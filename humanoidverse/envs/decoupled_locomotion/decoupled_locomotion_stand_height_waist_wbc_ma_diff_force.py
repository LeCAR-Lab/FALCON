from isaacgym.torch_utils import *
from humanoidverse.utils.torch_utils import (
    generate_sphere_sample_params,
    apply_sphere_sample_to_segments,
    sample_3d_directions,
)

import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma import LeggedRobotDecoupledLocomotionStanceHeightWBC

from isaac_utils.rotations import (
    my_quat_rotate,
)
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceHeightWBCForce(LeggedRobotDecoupledLocomotionStanceHeightWBC):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.left_hand_link = config.robot.force_control.left_hand_link
        self.right_hand_link = config.robot.force_control.right_hand_link
        self.left_hand_link_index = self.body_names.index(self.left_hand_link)
        self.right_hand_link_index = self.body_names.index(self.right_hand_link)
        logger.info(f"Left Hand Link: {self.left_hand_link}, Index: {self.left_hand_link_index}")
        logger.info(f"Right Hand Link: {self.right_hand_link}, Index: {self.right_hand_link_index}")

        self.left_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.left_ankle_dof_names]
        self.right_ankle_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.right_ankle_dof_names]

        self.j_left_ee = torch.zeros((self.num_envs, 6, 6+self.num_dofs), device=self.device)
        self.j_right_ee = torch.zeros((self.num_envs, 6, 6+self.num_dofs), device=self.device)
        
        self.zero_tapping_xy_force = self.config.get("zero_tapping_xy_force", False)
        self.joint_effort_limit_scale = self.config.robot.get("dof_effort_limit_scale", 1.0)
        self.joint_effort_limit = torch.tensor(self.config.robot.dof_effort_limit_list[:], 
                                               device=self.device, requires_grad=False) * self.joint_effort_limit_scale

    def _init_curriculum_scales(self):
        super()._init_curriculum_scales()
        if self.config.rewards.get("force_scale_curriculum", False):
            self.apply_force_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * self.config.rewards.force_scale_initial_scale
        else:
            self.apply_force_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _init_force_settings(self):
        # force initialization of value and pos
        self.left_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_ee_apply_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.apply_force_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.apply_force_pos_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # force position settings
        self.apply_force_pos_ratio_range = self.config.get("apply_force_pos_ratio_range", [0.0, 1.0])
        self.left_ee_apply_force_pos_ratio = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
        self.right_ee_apply_force_pos_ratio = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
        # force application settings
        self.only_apply_z_force_when_walking = self.config.get("only_apply_z_force_when_walking", True)
        self.only_apply_resistance_when_walking = self.config.get("only_apply_resistance_when_walking", True)
        # force ranges
        self.force_xyz_scale = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0], device=self.device)).sample((self.num_envs, ))
        self.left_force_xyz_scale = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0], device=self.device)).sample((self.num_envs, ))
        self.right_force_xyz_scale = torch.distributions.Dirichlet(torch.tensor([1.0, 1.0, 1.0], device=self.device)).sample((self.num_envs, ))
        self.force_range_low = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_high = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_range_low[:, 0] = self.config.apply_force_x_range[0]; self.force_range_high[:, 0] = self.config.apply_force_x_range[1]
        self.force_range_low[:, 1] = self.config.apply_force_y_range[0]; self.force_range_high[:, 1] = self.config.apply_force_y_range[1]
        self.force_range_low[:, 2] = self.config.apply_force_z_range[0]; self.force_range_high[:, 2] = self.config.apply_force_z_range[1]
        # force duration
        self.apply_force_duration = torch.randint(self.config.randomize_force_duration[0], self.config.randomize_force_duration[1] + 1, (self.num_envs, 1), device=self.device)
        self.left_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase = torch.rand((self.num_envs, 1), device=self.device)
        self.left_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        self.right_ee_apply_force_phase_ts = torch.zeros((self.num_envs, 1), device=self.device)
        # zero force probability
        self.left_zero_force_prob = self.config.get("zero_force_prob", [0.2, 0.2, 0.2])
        self.right_zero_force_prob = self.config.get("zero_force_prob", [0.2, 0.2, 0.2])
        if isinstance(self.left_zero_force_prob, float):
            self.left_zero_force_prob = [self.left_zero_force_prob] * 3
        if isinstance(self.right_zero_force_prob, float):
            self.right_zero_force_prob = [self.right_zero_force_prob] * 3
        self.left_zero_force_prob = torch.tensor(self.left_zero_force_prob, device=self.device)
        self.right_zero_force_prob = torch.tensor(self.right_zero_force_prob, device=self.device)
        self.left_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.left_zero_force_prob).float()
        self.right_zero_force = (torch.rand((self.num_envs, 3), device=self.device) < self.right_zero_force_prob).float()
        # random force probability
        self.random_force_prob = self.config.get("random_force_prob", 0.0)
        self.random_force = (torch.rand((self.num_envs, 1), device=self.device) < self.random_force_prob).float()
        # random force pos params
        self.left_fp_unit_dirs = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_fp_radius_scales = torch.zeros((self.num_envs, 1), device=self.device)
        self.right_fp_unit_dirs = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_fp_radius_scales = torch.zeros((self.num_envs, 1), device=self.device)
        # low pass filter for applied force
        self.use_lpf = self.config.get("use_lpf", False)
        self.filtered_left_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_left_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_right_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        self.force_filter_alpha = self.config.get("force_filter_alpha", 0.2)

        self.extend_curr_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.max_force_estimation = self.config.get("max_force_estimation", True)
        self.update_apply_force_phase = self.config.get("update_apply_force_phase", False)

    def _init_buffers(self):
        super()._init_buffers()
        # force settings
        self._init_force_settings()
        # stance/random force environments
        self.env_ids_stance = torch.where(self.commands[:, 4] == 0)[0]

        self.upper_body_tracking_sigma = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * \
                                         (self.config.rewards.upper_body_tracking_sigma_initial_scale \
                                         if self.config.rewards.get("upper_body_tracking_sigma_curriculum", False) \
                                         else self.config.rewards.reward_tracking_sigma.upper_body_dofs)
        
        self.upper_body_dofs_tracking_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.walking_env_mask = (self.commands[:, 4] == 1)
        zero_cmd_mask = torch.all(self.commands[:, :2] == 0, dim=-1) 
        self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask
        self.num_tapping_env = torch.sum(self.tapping_env_mask).item()
        # Generate a (num_tapping_env, 1) tensor with 50% probability of [1, 0] and 50% probability of [-1, 0]
        random_choices = torch.randint(0, 2, (self.num_envs, 1), device=self.device)  # Random 0 or 1
        self.random_tapping_dir = torch.where(random_choices == 0, torch.tensor([1, 0], device=self.device), torch.tensor([-1, 0], device=self.device))

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.update_apply_force_phase: self._update_apply_force_phase()
    
    def _update_apply_force_phase(self):
        # For the stance environments, update the phase timestamp
        self.left_ee_apply_force_phase_ts[self.env_ids_stance] += 1.0 / self.apply_force_duration[self.env_ids_stance]
        self.right_ee_apply_force_phase_ts[self.env_ids_stance] += 1.0 / self.apply_force_duration[self.env_ids_stance]
        # self.apply_force_phase_ts[self.env_ids_stance] += self.force_phase_ts_up_or_down[self.env_ids_stance] / self.apply_force_duration[self.env_ids_stance]
        self.left_ee_apply_force_phase[self.env_ids_stance] = abs(torch.remainder(self.left_ee_apply_force_phase_ts[self.env_ids_stance], 2.0) - 1.0)
        self.right_ee_apply_force_phase[self.env_ids_stance] = abs(torch.remainder(self.right_ee_apply_force_phase_ts[self.env_ids_stance], 2.0) - 1.0)

    def _calculate_max_ee_forces(self):
        # Apply the force at the hand links
        # Compute the linear z-axis jacobian for the left and right hand links
        jacobian = self.simulator.jacobian # Shape: (num_envs, num_bodies, 6, 35)
        self.j_left_ee = jacobian[:, self.left_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
        self.j_right_ee = jacobian[:, self.right_hand_link_index, :, :] # Shape: (num_envs, 6, 35)
        j_left_ee_joint_linear = self.j_left_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
        j_right_ee_joint_linear = self.j_right_ee[:, :3, 6:]  # Shape: (num_envs, 3, 29)
        j_left_ee_arm_joint_linear = j_left_ee_joint_linear[:, :, self.left_arm_dof_indices]  # Shape: (num_envs, 3, 7)
        j_right_ee_arm_joint_linear = j_right_ee_joint_linear[:, :, self.right_arm_dof_indices]  # Shape: (num_envs, 3, 7)
        j_left_ee_waist_joint_linear = j_left_ee_joint_linear[:, :, self.waist_dof_indices]  # Shape: (num_envs, 3, 3)
        j_right_ee_waist_joint_linear = j_right_ee_joint_linear[:, :, self.waist_dof_indices]  # Shape: (num_envs, 3, 3)
        joint_effort_est = self.ref_pd_torques # Shape: (num_envs, 35)
        # print(f"joint_effort_est: {joint_effort_est}")
        joint_effort_est *= 0.0
        max_delta_joint_effort = self.joint_effort_limit - joint_effort_est
        min_delta_joint_effort = -self.joint_effort_limit - joint_effort_est
        
        # First do the value, leave the direction for the scaling
        # Yuanhang: Apply X-Y-Z forces to the stance env, and apply only Z force to the walking env
        left_ee_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        left_ee_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        right_ee_force_min = torch.zeros((self.num_envs, 3), device=self.device)
        right_ee_force_max = torch.zeros((self.num_envs, 3), device=self.device)
        
        left_ee_force_max_delta_joint = torch.mul((1.0 / (torch.abs(j_left_ee_arm_joint_linear)+1.0e-2)), torch.stack([max_delta_joint_effort[:, self.left_arm_dof_indices]] * 3, dim=1)) # Shape: (num_envs, 3, 7)
        left_ee_force_min_delta_joint = torch.mul((1.0 / (torch.abs(j_left_ee_arm_joint_linear)+1.0e-2)), torch.stack([min_delta_joint_effort[:, self.left_arm_dof_indices]] * 3, dim=1))
        right_ee_force_max_delta_joint = torch.mul((1.0 / (torch.abs(j_right_ee_arm_joint_linear)+1.0e-2)), torch.stack([max_delta_joint_effort[:, self.right_arm_dof_indices]] * 3, dim=1)) # Shape: (num_envs, 3, 7)
        right_ee_force_min_delta_joint = torch.mul((1.0 / (torch.abs(j_right_ee_arm_joint_linear)+1.0e-2)), torch.stack([min_delta_joint_effort[:, self.right_arm_dof_indices]] * 3, dim=1))
        left_ee_force_min = torch.max(left_ee_force_min_delta_joint, dim=2)[0]
        left_ee_force_max = torch.min(left_ee_force_max_delta_joint, dim=2)[0]
        right_ee_force_min = torch.max(right_ee_force_min_delta_joint, dim=2)[0]
        right_ee_force_max = torch.min(right_ee_force_max_delta_joint, dim=2)[0]
        left_ee_force_min = torch.cat([left_ee_force_min[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        left_ee_force_min[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        left_ee_force_min[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        left_ee_force_max = torch.cat([left_ee_force_max[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        left_ee_force_max[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        left_ee_force_max[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        right_ee_force_min = torch.cat([right_ee_force_min[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        right_ee_force_min[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        right_ee_force_min[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        right_ee_force_max = torch.cat([right_ee_force_max[:, 0:1] * self.force_xyz_scale[:, 0:1],
                                        right_ee_force_max[:, 1:2] * self.force_xyz_scale[:, 1:2],
                                        right_ee_force_max[:, 2:3] * self.force_xyz_scale[:, 2:3]], dim=1)
        
        if self.only_apply_z_force_when_walking:
            walk_envs_idx = torch.where(self.commands[:, 4] == 1)[0]
            if len(walk_envs_idx) > 0:
                idx = walk_envs_idx
                left_ee_force_max[idx] = torch.cat([torch.zeros_like(left_ee_force_max[idx, 0:1]),
                                                    torch.zeros_like(left_ee_force_max[idx, 1:2]),
                                                    left_ee_force_max[idx, 2:3]], dim=1)
                left_ee_force_min[idx] = torch.cat([torch.zeros_like(left_ee_force_min[idx, 0:1]),
                                                    torch.zeros_like(left_ee_force_min[idx, 1:2]),
                                                    left_ee_force_min[idx, 2:3]], dim=1)
                right_ee_force_max[idx] = torch.cat([torch.zeros_like(right_ee_force_max[idx, 0:1]),
                                                     torch.zeros_like(right_ee_force_max[idx, 1:2]),
                                                     right_ee_force_max[idx, 2:3]], dim=1)
                right_ee_force_min[idx] = torch.cat([torch.zeros_like(right_ee_force_min[idx, 0:1]),
                                                     torch.zeros_like(right_ee_force_min[idx, 1:2]),
                                                     right_ee_force_min[idx, 2:3]], dim=1)

        # if self.is_evaluating: print(f"Max EE Force, Z: {ee_force_max_z}")
        # print(f"Max EE Force: {left_ee_force_max}, Min EE Force: {left_ee_force_min}, Force XYZ Scale: {self.left_force_xyz_scale}")

        if self.use_lpf:
            self.filtered_left_force_min = self.force_filter_alpha * left_ee_force_min + (1 - self.force_filter_alpha) * self.filtered_left_force_min
            self.filtered_left_force_max = self.force_filter_alpha * left_ee_force_max + (1 - self.force_filter_alpha) * self.filtered_left_force_max
            self.filtered_right_force_min = self.force_filter_alpha * right_ee_force_min + (1 - self.force_filter_alpha) * self.filtered_right_force_min
            self.filtered_right_force_max = self.force_filter_alpha * right_ee_force_max + (1 - self.force_filter_alpha) * self.filtered_right_force_max
            left_ee_force_min = self.filtered_left_force_min
            left_ee_force_max = self.filtered_left_force_max
            right_ee_force_min = self.filtered_right_force_min
            right_ee_force_max = self.filtered_right_force_max
        return left_ee_force_min, left_ee_force_max, right_ee_force_min, right_ee_force_max, j_left_ee_waist_joint_linear, j_right_ee_waist_joint_linear
    
    def _scale_forces(self, 
                      left_ee_force_min, left_ee_force_max, 
                      right_ee_force_min, right_ee_force_max,
                      j_left_ee_waist_joint_linear, j_right_ee_waist_joint_linear):
        # Apply different force phase to left and right EE
        # left_ee_force_phased = self.force_range_low + (self.force_range_high - self.force_range_low) * self.left_ee_apply_force_phase
        # right_ee_force_phased = self.force_range_low + (self.force_range_high - self.force_range_low) * self.right_ee_apply_force_phase
        left_ee_force_phased = left_ee_force_min + (left_ee_force_max - left_ee_force_min) * self.left_ee_apply_force_phase
        right_ee_force_phased = right_ee_force_min + (right_ee_force_max - right_ee_force_min) * self.right_ee_apply_force_phase
        
        # Apply the force scale and the force phase to the maximum force norm
        left_hand_force = left_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)
        right_hand_force = right_ee_force_phased * self.apply_force_scale + torch.rand((self.num_envs, 3), device=self.device) # Shape: (num_envs, 3)

        # Zero the force if zero force probability is met
        left_hand_force *= (1 - self.left_zero_force)
        right_hand_force *= (1 - self.right_zero_force)

        # Clip using the min/max as bounds
        left_hand_force = torch.clip(left_hand_force, self.force_range_low, self.force_range_high)
        right_hand_force = torch.clip(right_hand_force, self.force_range_low, self.force_range_high)

        # Apply waist joint constraint as post-processing
        # if j_left_ee_waist_joint_linear is not None and j_right_ee_waist_joint_linear is not None:
        #     left_ee_torque_on_waist = j_left_ee_waist_joint_linear.transpose(1, 2).bmm(left_hand_force.unsqueeze(-1)).squeeze(-1)
        #     right_ee_torque_on_waist = j_right_ee_waist_joint_linear.transpose(1, 2).bmm(right_hand_force.unsqueeze(-1)).squeeze(-1)
        #     total_waist_torque = left_ee_torque_on_waist + right_ee_torque_on_waist  
        #     scaling_factor = torch.min(torch.ones_like(total_waist_torque), 
        #                             self.joint_effort_limit[self.waist_dof_indices] / (total_waist_torque.abs() + 1e-6))
        #     if scaling_factor.min() < 1.0 and self.is_evaluating:
        #         print(f"Scaling Factor: {scaling_factor.min()}")
        #     left_hand_force *= scaling_factor
        #     right_hand_force *= scaling_factor

        return left_hand_force, right_hand_force

    def _calculate_ee_forces(self):
        if self.max_force_estimation:
            left_ee_force_min, left_ee_force_max, \
            right_ee_force_min, right_ee_force_max, \
            j_left_ee_waist_joint_linear, j_right_ee_waist_joint_linear = self._calculate_max_ee_forces()
        else:
            left_ee_force_min = right_ee_force_min = self.force_range_low
            left_ee_force_max = right_ee_force_max = self.force_range_high
            j_left_ee_waist_joint_linear = j_right_ee_waist_joint_linear = None
        # print(f"Left EE Force Min: {left_ee_force_min}, Max: {left_ee_force_max}")
        # print(f"Right EE Force Min: {right_ee_force_min}, Max: {right_ee_force_max}")
        left_hand_force, right_hand_force = self._scale_forces(left_ee_force_min, left_ee_force_max, 
                                                               right_ee_force_min, right_ee_force_max,
                                                               j_left_ee_waist_joint_linear, 
                                                               j_right_ee_waist_joint_linear)

        if self.is_evaluating:
            self.walking_env_mask = (self.commands[:, 4] == 1)
            zero_cmd_mask = torch.all(self.commands[:, :2] == 0, dim=-1) 
            self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask
            self.num_tapping_env = torch.sum(self.tapping_env_mask).item()
            # left_hand_force = torch.zeros_like(left_hand_force)
            # right_hand_force = torch.zeros_like(right_hand_force)
            # left_hand_force[:, 0] = -30
            # right_hand_force[:, 0] = -30
            pass
        
        # Get the walking direction
        walking_dir = quat_rotate(self.base_quat, 
                                  torch.cat([self.commands[:, :2], 
                                             torch.zeros((self.num_envs, 1), device=self.device)], 
                                             dim=-1))[:, :2]  # (batch_size, 2)
        walking_dir_norm = torch.norm(walking_dir, dim=-1, keepdim=True) + 1e-6 
        walking_dir[self.tapping_env_mask] = quat_rotate(self.base_quat[self.tapping_env_mask],
                                                         torch.cat([self.random_tapping_dir[self.tapping_env_mask],
                                                         torch.zeros((self.num_tapping_env, 1), device=self.device)],
                                                         dim=-1))[:, :2]
        walking_dir_norm[self.tapping_env_mask] = torch.norm(walking_dir[self.tapping_env_mask], dim=-1, keepdim=True) + 1e-6
        if self.zero_tapping_xy_force:
            walking_dir[self.tapping_env_mask] *= 0.0
        if self.only_apply_resistance_when_walking:
            walking_dir_unit = torch.zeros_like(walking_dir)
            walking_dir_unit[self.walking_env_mask] = -walking_dir[self.walking_env_mask] / walking_dir_norm[self.walking_env_mask]
            
            # Get the x-y components of the force
            left_hand_force_xy = left_hand_force[:, :2]  
            right_hand_force_xy = right_hand_force[:, :2]

            left_hand_force_proj = left_hand_force_xy.clone()
            right_hand_force_proj = right_hand_force_xy.clone()

            left_hand_force_proj[self.walking_env_mask] = torch.abs(torch.sum(left_hand_force_xy[self.walking_env_mask] * walking_dir_unit[self.walking_env_mask], 
                                                                              dim=-1, keepdim=True)) * walking_dir_unit[self.walking_env_mask]
            right_hand_force_proj[self.walking_env_mask] = torch.abs(torch.sum(right_hand_force_xy[self.walking_env_mask] * walking_dir_unit[self.walking_env_mask], 
                                                                               dim=-1, keepdim=True)) * walking_dir_unit[self.walking_env_mask]

            # Combine the x-y components with the z component
            left_hand_force = torch.cat([left_hand_force_proj, left_hand_force[:, 2:3]], dim=-1)
            right_hand_force = torch.cat([right_hand_force_proj, right_hand_force[:, 2:3]], dim=-1)

        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_hand_force.clone())
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_hand_force.clone())

        # Apply the force to the hand links
        self.apply_force_tensor[:, self.left_hand_link_index, :] = left_hand_force
        self.apply_force_tensor[:, self.right_hand_link_index, :] = right_hand_force
    
    def _apply_force_in_physics_step(self):
        # Apply the force in the physics step
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        
        if self.config.apply_force_in_physics_step:
            self.simulator.apply_rigid_body_force_at_pos_tensor(self.apply_force_tensor, self.apply_force_pos_tensor)
        
        self.simulator.apply_torques_at_dof(self.torques)
    
    def _physics_step(self):
        self.render()
        self._calculate_ee_forces()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()
    
    def _resample_force_settings(self, env_ids):
        # Yuanhang: use Dirichlet distribution to sample the force at maximum and minimum
        if env_ids.numel() > 0:  # Only update if env_ids is not empty
            self.left_force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
                torch.tensor([1.0, 1.0, 1.0], device=self.device)
            ).sample((len(env_ids), ))
            self.right_force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
                torch.tensor([1.0, 1.0, 1.0], device=self.device)
            ).sample((len(env_ids), ))
            self.force_xyz_scale[env_ids] = torch.distributions.Dirichlet(
                 torch.tensor([1.0, 1.0, 1.0], device=self.device)
             ).sample((len(env_ids), ))
        self.apply_force_duration[env_ids] = torch.randint(self.config.randomize_force_duration[0], 
                                                           self.config.randomize_force_duration[1] + 1, 
                                                           (len(env_ids), 1), device=self.device)
        self.left_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase_ts[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.left_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.right_ee_apply_force_phase[env_ids] = torch.rand((len(env_ids), 1), device=self.device)
        self.left_zero_force[env_ids] = (torch.rand((len(env_ids), 3), device=self.device) < self.left_zero_force_prob).float()
        self.right_zero_force[env_ids] = (torch.rand((len(env_ids), 3), device=self.device) < self.right_zero_force_prob).float()
        self.random_force[env_ids] = (torch.rand((len(env_ids), 1), device=self.device) < self.random_force_prob).float()

        # Resample the apply force sphere params
        self.left_fp_unit_dirs[env_ids], self.left_fp_radius_scales[env_ids] = generate_sphere_sample_params(len(env_ids), self.device)
        self.right_fp_unit_dirs[env_ids], self.right_fp_radius_scales[env_ids] = generate_sphere_sample_params(len(env_ids), self.device)
        self.left_ee_apply_force_pos_ratio[env_ids] = torch.rand((len(env_ids), 1), device=self.device) * (self.apply_force_pos_ratio_range[1] - self.apply_force_pos_ratio_range[0]) + self.apply_force_pos_ratio_range[0]
        self.right_ee_apply_force_pos_ratio[env_ids] = torch.rand((len(env_ids), 1), device=self.device) * (self.apply_force_pos_ratio_range[1] - self.apply_force_pos_ratio_range[0]) + self.apply_force_pos_ratio_range[0]

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        # Yuanhang: do not reset the envs, otherwise the episode lengths conflict with the ppo buffer
        # self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))
        # Yuanhang: reset the filtered forces because the motion is changed suddenly
        self.filtered_left_force_max *= 0.0
        self.filtered_left_force_min *= 0.0
        self.filtered_right_force_max *= 0.0
        self.filtered_right_force_min *= 0.0

    def _resample_motion_times(self, env_ids):
        # return
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # Yuanhang: reset the filtered forces because the motion is changed suddenly
        self.filtered_left_force_max[env_ids] *= 0.0
        self.filtered_left_force_min[env_ids] *= 0.0
        self.filtered_right_force_max[env_ids] *= 0.0
        self.filtered_right_force_min[env_ids] *= 0.0

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # Resample the force settings for the environments
        self._resample_force_settings(env_ids)
        
        self.walking_env_mask = (self.commands[:, 4] == 1)
        zero_cmd_mask = torch.all(self.commands[:, :2] == 0, dim=-1) 
        self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask
        self.num_tapping_env = torch.sum(self.tapping_env_mask).item()
        # Restrict to resample only for tapping envs within envs_id
        tapping_envs_to_resample = self.tapping_env_mask[env_ids]
        tapping_indices = env_ids[tapping_envs_to_resample]
        # Generate random directions for them
        random_choices = torch.randint(0, 2, (tapping_indices.shape[0], 1), device=self.device)  # Random 0 or 1
        self.random_tapping_dir[tapping_indices] = torch.where(random_choices == 0,
                                                               torch.tensor([1, 0], device=self.device),
                                                               torch.tensor([-1, 0], device=self.device))

        if len(env_ids) != 0:
            self.env_ids_stance = torch.where(self.commands[env_ids, 4] == 0)[0]
    
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        if self.config.get('analysis', False):
            self._resample_commands(torch.arange(self.num_envs, device=self.device))
        else:
            self.commands = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
            self.commands[:, 8] = self.config.rewards.desired_base_height
            if command is not None:
                self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

            # stance/random force environments
            self.env_ids_stance = torch.where(self.commands[:, 4] == 0)[0]

            self.left_ee_apply_force_pos_ratio = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
            self.right_ee_apply_force_pos_ratio = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * (self.apply_force_pos_ratio_range[0] + self.apply_force_pos_ratio_range[1]) / 2.0
            
            self.left_force_xyz_scale = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1).requires_grad_(False)
            self.right_force_xyz_scale = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1).requires_grad_(False)
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # Apply force scale
        force_level = self.config.get("eval_force_level", 0)
        if force_level == 0:
            self.apply_force_scale = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        elif force_level == 1:
            self.apply_force_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) / 2
        elif force_level == 2:
            self.apply_force_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Hardcode
        self.only_apply_z_force_when_walking = False
        self.only_apply_resistance_when_walking = True
        self.motion_start_times *= 0.0
        self.max_force_estimation = True
        self.update_apply_force_phase = True
        self.use_lpf = True
        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)
    
    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Resets the environments with the given ids and optionally to the target states
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.config.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        if self.config.rewards.upper_body_motion_scale_curriculum:
            self._update_upper_body_motion_scale_curriculum(env_ids)
        else:
            self.action_scale_upper_body[env_ids] = 1.0
        if self.config.rewards.get("force_scale_curriculum", False):
            self._update_force_scale_curriculum(env_ids)
        if self.config.rewards.get("command_height_scale_curriculum", False):
            self._update_command_height_curriculum(env_ids)
        else:
            self.command_height_scale[env_ids] = 1.0
        if self.config.rewards.get("upper_body_tracking_sigma_curriculum", False):
            self._update_upper_body_tracking_sigma_curriculum(env_ids)
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)
        
        self.upper_body_dofs_tracking_reward[env_ids] *= 0.0

        self.filtered_left_force_max[env_ids] *= 0.0
        self.filtered_left_force_min[env_ids] *= 0.0
        self.filtered_right_force_max[env_ids] *= 0.0
        self.filtered_right_force_min[env_ids] *= 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        B = self.motion_ids.shape[0]
        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        self.extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, self.extend_curr_pos], dim=1)
        self.marker_coords[:] = self._rigid_body_pos_extend.reshape(B, -1, 3) # visualize the robot rigid body pos
        # self.marker_coords[:] = self.ref_body_pos_extend.reshape(B, -1, 3) # visualize the target rigid body pos
        # Hardcode: decide which one to use
        # 1. Apply forces within randomly sample shperes
        # left_ee_apply_force_pos = apply_sphere_sample_to_segments(self.simulator._rigid_body_pos[:, self.left_hand_link_index, :],
        #                                                           self.extend_curr_pos[:, 0, :], 
        #                                                           self.left_fp_unit_dirs, self.left_fp_radius_scales)
        # right_ee_apply_force_pos = apply_sphere_sample_to_segments(self.simulator._rigid_body_pos[:, self.right_hand_link_index, :], 
        #                                                            self.extend_curr_pos[:, 1, :], 
        #                                                            self.right_fp_unit_dirs, self.right_fp_radius_scales)
        
        # 2. Apply force between the hand link and the sampled sphere
        left_ee_apply_force_pos = (self.extend_curr_pos[:, 0, :] - self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]) * self.left_ee_apply_force_pos_ratio + self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]
        right_ee_apply_force_pos = (self.extend_curr_pos[:, 1, :] - self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]) * self.right_ee_apply_force_pos_ratio + self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]
        # 3. Apply force at the hand links
        # self.apply_force_pos_tensor[:, self.left_hand_link_index,:] = self.simulator._rigid_body_pos[:, self.left_hand_link_index, :]
        # self.apply_force_pos_tensor[:, self.right_hand_link_index,:] = self.simulator._rigid_body_pos[:, self.right_hand_link_index, :]
        self.apply_force_pos_tensor[:, self.left_hand_link_index,:] = left_ee_apply_force_pos
        self.apply_force_pos_tensor[:, self.right_hand_link_index,:] = right_ee_apply_force_pos
        if self.is_evaluating:
            # self.ref_upper_dof_pos *= 0
            pass
    
    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            # for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
            #     if self.config.robot.motion.visualization.customize_color:
            #         color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
            #     else:
            #         color_inner = (0.3, 0.3, 0.3)
            #     color_inner = tuple(color_inner)

            #     self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id)
            # draw forces

            force_left_hand = self.apply_force_tensor[env_id, self.left_hand_link_index, :]
            force_right_hand = self.apply_force_tensor[env_id, self.right_hand_link_index, :]

            force_pos_left_hand = self.apply_force_pos_tensor[env_id, self.left_hand_link_index, :]
            force_pos_right_hand = self.apply_force_pos_tensor[env_id, self.right_hand_link_index, :]
            
            force_list = [force_left_hand, force_right_hand]
            pos_list = [force_pos_left_hand, force_pos_right_hand]
            force_mag_list = [0.025, 0.025]
            color_schems = [(0.851, 0.144, 0.07), (0.851, 0.144, 0.07)]
            # color_schems = [(0., .5, 0.), (0., .5, 0.)]
            line_widths = [0.02, 0.02]

            for force, pos, force_mag, color, line_width in zip(force_list, pos_list, force_mag_list, color_schems, line_widths):
                for _ in range(20):
                    start_point = pos + torch.rand(3, device=self.device) * line_width
                    end_point = pos + force * force_mag
                    self.simulator.draw_line(Point(start_point +torch.rand(3, device=self.device) * line_width),
                                        Point(end_point + torch.rand(3, device=self.device) * line_width),
                                        Point(color),
                                        env_id)

    ############################ Curriculum #############################
    def _update_upper_body_tracking_sigma_curriculum(self, env_ids):
        """ Implement upper body tracking sigma curriculum based on the tracking performance
        """
        upper_body_tracking_error = self.upper_body_dofs_tracking_reward[env_ids] / self.episode_length_buf[env_ids]
        env_ids_scale_up_mask = upper_body_tracking_error < self.config.rewards.upper_body_tracking_sigma_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = upper_body_tracking_error > self.config.rewards.upper_body_tracking_sigma_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.upper_body_tracking_sigma[env_ids_scale_up] += self.config.rewards.upper_body_tracking_sigma_scale_up
        self.upper_body_tracking_sigma[env_ids_scale_down] -= self.config.rewards.upper_body_tracking_sigma_scale_down
        # Clip the scale
        self.upper_body_tracking_sigma[env_ids] = torch.clip(self.upper_body_tracking_sigma[env_ids],
                                                             self.config.rewards.upper_body_tracking_sigma_min,
                                                             self.config.rewards.upper_body_tracking_sigma_max)


    
    def _update_force_scale_curriculum(self, env_ids):
        """ Implement force curriculum

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        env_ids_scale_up_mask = self.episode_length_buf[env_ids] > self.config.rewards.force_scale_up_threshold
        # env_ids_scale_up_mask = self.upper_body_dofs_tracking_reward[env_ids] / self.episode_length_buf[env_ids] > self.config.rewards.force_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = self.episode_length_buf[env_ids] < self.config.rewards.force_scale_down_threshold
        # env_ids_scale_down_mask = self.upper_body_dofs_tracking_reward[env_ids] / self.episode_length_buf[env_ids] < self.config.rewards.force_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.apply_force_scale[env_ids_scale_up] += self.config.rewards.force_scale_up
        self.apply_force_scale[env_ids_scale_down] -= self.config.rewards.force_scale_down
        # self.apply_force_scale[env_ids_scale_up] *= (1 + self.config.rewards.reward_penalty_degree)
        # self.apply_force_scale[env_ids_scale_down] *= (1 - self.config.rewards.reward_penalty_degree)
        # Clip the scale
        self.apply_force_scale[env_ids] = torch.clip(self.apply_force_scale[env_ids], 
                                                     self.config.rewards.force_scale_min, 
                                                     self.config.rewards.force_scale_max)

    ########################### FEET REWARDS ###########################
    
    def _reward_tracking_upper_body_dofs(self):
        # Reward the difference between the waist dof pos and the reference
        upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
        upper_body_dofs_error =  torch.sum(torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1)
        upper_body_dofs_tracking_reward =  torch.exp(-upper_body_dofs_error/(self.upper_body_tracking_sigma.squeeze(-1)))
        self.upper_body_dofs_tracking_reward += upper_body_dofs_tracking_reward
        return upper_body_dofs_tracking_reward

    def _reward_tracking_stance_base_height(self):
        # Tracking of base height commands (z axe)
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2])
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height) * (1 - self.commands[:, 4]) # only apply the base height penalty if stance
    
    def _reward_tracking_walk_base_height(self):
        # Tracking of base height commands (z axe)
        total_apply_force = torch.norm(self.apply_force_tensor[:, self.left_hand_link_index, :] + self.apply_force_tensor[:, self.right_hand_link_index, :], dim=1)
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2]) * (1 - torch.clip(total_apply_force / 50, 0, 1))
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height) * self.commands[:, 4]

    def _reward_penalty_ankle_roll(self):
        # Compute the penalty for ankle roll
        left_ankle_roll = self.simulator.dof_pos[:, self.left_ankle_dof_indices[1:2]]
        right_ankle_roll = self.simulator.dof_pos[:, self.right_ankle_dof_indices[1:2]]
        return torch.sum(torch.abs(left_ankle_roll) + torch.abs(right_ankle_roll), dim=1) # * (1 - self.commands[:, 4]) # only apply the ankle roll penalty if stance
    
    def _reward_penalty_stance_feet_vel(self):
        # Penalize the velocity of the stance feet
        left_ee_lin_vel = self.simulator._rigid_body_vel[:, self.left_hand_link_index, 0:3]
        left_ee_ang_vel = self.simulator._rigid_body_ang_vel[:, self.left_hand_link_index, 0:3]
        left_ee_vel = torch.cat([left_ee_lin_vel, left_ee_ang_vel], dim=1)
        right_ee_lin_vel = self.simulator._rigid_body_vel[:, self.right_hand_link_index, 0:3]
        right_ee_ang_vel = self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, 0:3]
        right_ee_vel = torch.cat([right_ee_lin_vel, right_ee_ang_vel], dim=1)
        return (torch.norm(left_ee_vel, dim=1) + torch.norm(right_ee_vel, dim=1)) * (1 - self.commands[:, 4])
    
    def _reward_penalty_ee_lin_acc(self):
        # Penalize the end effector acceleration
        left_ee_lin_acc = self.simulator._rigid_body_vel[:, self.left_hand_link_index, 0:3] - self.last_left_ee_vel
        right_ee_lin_acc = self.simulator._rigid_body_vel[:, self.right_hand_link_index, 0:3] - self.last_right_ee_vel
        end_effector_acc = torch.cat([left_ee_lin_acc.unsqueeze(1), right_ee_lin_acc.unsqueeze(1)], dim=1)
        end_effector_acc_norm = torch.norm(end_effector_acc, dim=2)
        return torch.sum(end_effector_acc_norm, dim=1)

    def _reward_penalty_ee_ang_acc(self):
        # Penalize the end effector angular acceleration
        left_ee_ang_acc = self.simulator._rigid_body_ang_vel[:, self.left_hand_link_index, 0:3] - self.last_left_ee_ang_vel
        right_ee_ang_acc = self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, 0:3] - self.last_right_ee_ang_vel
        end_effector_ang_acc = torch.cat([left_ee_ang_acc.unsqueeze(1), right_ee_ang_acc.unsqueeze(1)], dim=1)
        end_effector_ang_acc_norm = torch.norm(end_effector_ang_acc, dim=2)
        return torch.sum(end_effector_ang_acc_norm, dim=1)

    ######################### Observations #########################
    def _get_obs_left_ee_apply_force(self):
        # return the force exerted on the left ee (hand)
        # print(f"Apply Force Tensor: {self.apply_force_tensor[:, self.left_hand_link_index, 2:3]}")
        return self.left_ee_apply_force
    
    def _get_obs_right_ee_apply_force(self):
        # return the force exerted on the right ee (hand)
        return self.right_ee_apply_force
    