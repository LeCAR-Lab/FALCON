from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from humanoidverse.envs.env_utils.general import class_to_dict
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase

from scipy.stats import vonmises


class LeggedRobotLocomotion(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._init_gait_params()
        self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
        self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
        self.upper_left_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
        self.upper_right_arm_dof_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]
        self.init_done = True

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        self.stance_base_height_penalty_scale = self.config.rewards.get("stance_base_height_penalty_scale", 1.0)

    def _init_gait_params(self):
        # Initialize the normalized period of the swing phase
        self.a_swing = 0.0 # start of the swing phase
        self.b_swing = 0.5 # end of the swing phase
        self.a_stance = 0.5 # start of the stance phase
        self.b_stance = 1.0 # end of the stance phase
        self.kappa = 4.0 # shared variance in Von Mises 
        self.left_offset = 0.0 # left foot offset
        self.right_offset = 0.5 # right foot offset
        
        self.left_feet_height = torch.zeros(self.num_envs, device=self.device) # left feet height
        self.right_feet_height = torch.zeros(self.num_envs, device=self.device) # right feet height
        
        self.phase_time = torch.zeros(self.num_envs, dtype=torch.float32, requires_grad=False, device=self.device)
        self.phase_time_np = np.zeros(self.num_envs, dtype=np.float32)
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        # Initialize the gait period
        if hasattr(self.config.rewards, "gait_period"):
            if not self.config.rewards.gait_period:
                self.T = self.config.rewards.gait_period # gait period in seconds # gait period in seconds
            else:
                self.T = 1. # gait period in seconds
        else:
            self.T = 1.
        
        if hasattr(self.config.obs, "use_phase"):
            # Randomize the gait phase time
            if self.config.obs.use_phase:
                self.phi_offset = np.random.rand(self.num_envs)*self.T
            else: self.phi_offset = np.zeros(self.num_envs)
        else:
            self.phi_offset = np.zeros(self.num_envs)
        # Initialize the target arm joint positions
        self.swing_arm_joint_pos = torch.tensor([-1.04, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        self.stance_arm_joint_pos = torch.tensor([0.757, 0.0, 0.0, 1.57,
                                                0.0, 0.0, 0.0], device=self.device, dtype=torch.float, requires_grad=False)
        print("phi_offset: ", self.phi_offset)
    

    def _setup_simulator_control(self):
        self.simulator.commands = self.commands

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
    
    def _post_physics_step(self):
        super()._post_physics_step()
        self.update_phase_time()

    def update_phase_time(self):
        # Update the phase time
        self.phase_time_np = self._calc_phase_time()
        self.phase_time = torch.tensor(self.phase_time_np, device=self.device, dtype=torch.float, requires_grad=False)
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        if not self.is_evaluating:
            self._resample_commands(env_ids)

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

    ########################### TRACKING REWARDS ###########################

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.config.rewards.reward_tracking_sigma.ang_vel)

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_penalty_ang_vel_xy_torso(self):
        # Penalize xy axes base angular velocity

        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)
    

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    ########################### FEET REWARDS ###########################

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # Record the air time of the first contact
        self.last_feet_air_time[first_contact] = self.feet_air_time[first_contact]
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_penalty_diff_feet_air_time(self):
        # Reward symmetric feet air time
        return torch.abs(self.last_feet_air_time[:, 0] - self.last_feet_air_time[:, 1])

    def _reward_penalty_in_the_air(self):
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_foot_contact = contact_filt[:,0]
        second_foot_contact = contact_filt[:,1]
        reward = ~(first_foot_contact | second_foot_contact)
        return reward



    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)


    def _reward_penalty_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_base_height(self):
        # Penalize base height away from target

        base_height = self.simulator.robot_root_states[:, 2]
        return torch.square(base_height - self.config.rewards.desired_base_height)

    def _reward_feet_heading_alignment(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])


        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_feet_ori(self):
        left_quat = self.simulator._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self.simulator._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_penalty_feet_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:,self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values # [num_env], # select the foot closer to target 
        return torch.clip(dif - 0.02, min=0.) # target - 0.02 ~ target + 0.02 is acceptable 
    
    def _reward_penalty_feet_swing_height(self):
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        height_error = torch.square(self.simulator._rigid_body_pos[:, self.feet_indices, 2] - \
                                    self.config.rewards.feet_height_target) * ~contact
        return torch.sum(height_error, dim=(1))
    
    def _reward_penalty_close_feet_xy(self):
        # returns 1 if two feet are too close
        left_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[0], :2]
        right_foot_xy = self.simulator._rigid_body_pos[:, self.feet_indices[1], :2]
        feet_distance_xy = torch.norm(left_foot_xy - right_foot_xy, dim=1)
        return (feet_distance_xy < self.config.rewards.close_feet_threshold) * 1.0
    
    def _reward_penalty_far_feet_base_y(self):
        # returns 1 if two feet are too close in the y-axis
        left_foot_xyz = self.simulator._rigid_body_pos[:, self.feet_indices[0], :]
        right_foot_xyz = self.simulator._rigid_body_pos[:, self.feet_indices[1], :]
        feet_distance_xyz = torch.abs(left_foot_xyz - right_foot_xyz)
        base_feet_distance_xyz = quat_rotate_inverse(self.base_quat, feet_distance_xyz)
        return (base_feet_distance_xyz[:, 1] > self.config.rewards.close_feet_base_y_threshold) * self.commands[:, 4] # only apply when walking 

    def _reward_penalty_close_knees_xy(self):
        # returns 1 if two knees are too close
        left_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[0], :2]
        right_knee_xy = self.simulator._rigid_body_pos[:, self.knee_indices[1], :2]
        self.knee_distance_xy = torch.norm(left_knee_xy - right_knee_xy, dim=1)
        return (self.knee_distance_xy < self.config.rewards.close_knees_threshold)* 1.0
    

    def _reward_upperbody_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:,self.upper_dof_indices])
        return torch.sum(deviation, dim=1)
    
    def _reward_penalty_shift_in_zero_command(self):
        shift_vel = torch.norm(self.simulator._rigid_body_vel[:, 0, :2], dim=-1) * (torch.norm(self.commands[:, :2], dim=1) < 0.2) * self.commands[:, 4]
        # print(shift_vel)
        return shift_vel

    def _reward_penalty_ang_shift_in_zero_command(self):
        ang_vel = torch.abs(self.simulator._rigid_body_ang_vel[:, 0, 2])  # assuming index 5 = angular z
        # Apply penalty only when there's no angular command (or very low)
        zero_ang_command_mask = (torch.abs(self.commands[:, 2]) < 0.1)
        ang_shift = ang_vel * zero_ang_command_mask * self.commands[:, 4]
        return ang_shift

    ########################### GAIT REWARDS ###########################
    def _calc_phase_time(self):
        # Calculate the phase time
        episode_length_np = self.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
        return phase_time

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2): # left and right feet
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def calculate_phase_expectation(self, phi, offset=0, phase="swing"):
        """
        Calculate the expectation value of I_i(φ).

        Parameters:
        phi (float): The given phase time.
        offset (float): The offset of the phase time.

        Returns:
        float: The expectation value of I_i(φ).
        """
        # print("phase_time: ", phi)
        phi = (phi + offset) % 1
        phi *= 2 * np.pi
        # Create Von Mises distribution objects for A_i and B_i
        if phase == "swing":
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_swing)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_swing)
        else:
            dist_A = vonmises(self.kappa, loc=2 * np.pi * self.a_stance)
            dist_B = vonmises(self.kappa, loc=2 * np.pi * self.b_stance)
        # Calculate P(A_i < φ) and P(B_i < φ)
        P_A_less_phi = dist_A.cdf(phi)
        P_B_less_phi = dist_B.cdf(phi)
        # Calculate P(A_i < φ < B_i)
        P_A_phi_B = P_A_less_phi * (1 - P_B_less_phi)
        # Calculate the expectation value of I_i
        E_I_i = P_A_phi_B

        return E_I_i
    
    def _reward_gait_period(self):
        """
        Jonah Siekmann, et al. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
        paper link: https://arxiv.org/abs/2011.01387
        """
        # Calculate the expectation value of I_i of left and right feet
        E_I_l_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="swing")
        E_I_l_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.left_offset, phase="stance")
        E_I_r_swing = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="swing")
        E_I_r_stance = self.calculate_phase_expectation(self.phase_time_np, offset=self.right_offset, phase="stance")
        # print("E_I_l_swing: ", E_I_l_swing, ", E_I_r_swing: ", E_I_r_swing)
        # print("E_I_l_stance: ", E_I_l_stance, ", E_I_r_stance: ", E_I_r_stance)
        ## Convert to tensor
        E_I_l_swing = torch.tensor(E_I_l_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_swing = torch.tensor(E_I_r_swing, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_l_stance = torch.tensor(E_I_l_stance, device=self.device, dtype=torch.float, requires_grad=False)
        E_I_r_stance = torch.tensor(E_I_r_stance, device=self.device, dtype=torch.float, requires_grad=False)
        # Get the contact forces and velocities of the feet, and the velocities of the arm ee
        Ff_left = torch.norm(self.simulator.contact_forces[:, self.feet_indices[0], :], dim=-1) # left foot contact force
        Ff_right = torch.norm(self.simulator.contact_forces[:, self.feet_indices[1], :], dim=-1) # right foot contact force
        vf_left = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[0], :], dim=-1) # left foot velocity
        vf_right = torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices[1], :], dim=-1) # right foot velocity
        # print("Ff_left: ", Ff_left, ", Ff_right: ", Ff_right)
        # print("vf_left: ", vf_left, ", vf_right: ", vf_right)
        reward_gait = E_I_l_swing * torch.exp(-Ff_left**2) + E_I_r_swing * torch.exp(-Ff_right**2) + \
                      E_I_l_stance * torch.exp(-200*vf_left**2) + E_I_r_stance * torch.exp(-200*vf_right**2)
        # Sum up the gait reward
        return reward_gait

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity x commands
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_x_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity y commands
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_penalty_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.simulator.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return self.commands[:, :2]
    
    def _get_obs_command_ang_vel(self):
        return self.commands[:, 2:3]
    
    def _get_obs_phase_time(self):
        return self.phase_time.unsqueeze(1)
    
    def _get_obs_sin_phase(self):
        return torch.sin(2 * np.pi * self.phase_time).unsqueeze(1)
    
    def _get_obs_cos_phase(self):
        return torch.cos(2 * np.pi * self.phase_time).unsqueeze(1)