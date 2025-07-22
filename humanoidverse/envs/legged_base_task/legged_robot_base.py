import numpy as np

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from isaac_utils.rotations import get_euler_xyz_in_tensor
from humanoidverse.envs.base_task.base_task import BaseTask

from termcolor import colored
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
import copy

class LeggedRobotBase(BaseTask):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._domain_rand_config()
        self._prepare_reward_function()
        self.is_evaluating = False
        self.init_done = True

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        # self.simulator.dof_pos = self.simulator.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # self.simulator.dof_vel = self.simulator.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        if self.config.robot.get("force_control", None) is not None:
            self.left_hand_link = self.config.robot.force_control.left_hand_link
            self.right_hand_link = self.config.robot.force_control.right_hand_link
            self.left_hand_link_index = self.body_names.index(self.left_hand_link)
            self.right_hand_link_index = self.body_names.index(self.right_hand_link)
        logger.info(f"Left Hand Link: {self.left_hand_link}, Index: {self.left_hand_link_index}")
        logger.info(f"Right Hand Link: {self.right_hand_link}, Index: {self.right_hand_link_index}")

        self.base_quat = self.simulator.base_quat
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # initialize some data used later on
        self._init_counters()
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_after_delay = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(self.simulator.robot_root_states[:, 7:13])
        self.last_left_ee_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_left_ee_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_right_ee_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_right_ee_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._init_domain_rand_buffers()

        # for reward penalty curriculum
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            self.push_interval_s = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (self.num_envs,), device=self.device)

    def _init_counters(self):
        self.common_step_counter = 0
        self.push_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.push_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)

    def _update_counters_each_step(self):
        self.common_step_counter  +=1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        self.lower_body_ctrl_delay = self.config.domain_rand.get("lower_body_ctrl_delay", False)
        self.upper_body_ctrl_delay = self.config.domain_rand.get("upper_body_ctrl_delay", False)
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.config.domain_rand.ctrl_delay_step_range[1]+1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                                self.config.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        # self._link_mass_scale = torch.ones(self.num_envs, len(self.config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_vel_buf = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(colored(f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}", "green"))
        
        self.reward_scales = self.config.rewards.reward_scales
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        self.use_reward_penalty_curriculum = self.config.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.config.rewards.reward_initial_penalty_scale

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}")

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
            # reward episode sums
            self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                for name in self.reward_scales.keys()}

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]
        # actions *= 0.0
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            # Delay the selected body actions
            # actions_after_delay = self.actions.clone()
            # if self.lower_body_ctrl_delay:
            #     # print("Applying lower body action delay")
            #     actions_after_delay[:, self.lower_dof_indices] = \
            #         self.action_queue[torch.arange(self.num_envs), self.action_delay_idx][:, self.lower_dof_indices]
            # if self.upper_body_ctrl_delay:
            #     # print("Applying upper body action delay")
            #     actions_after_delay[:, self.upper_dof_indices] = \
            #         self.action_queue[torch.arange(self.num_envs), self.action_delay_idx][:, self.upper_dof_indices]
            # self.actions_after_delay = actions_after_delay
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()


    def _physics_step(self):
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        self.simulator.apply_torques_at_dof(self.torques)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
    
    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _update_tasks_callback(self):
        # push robots
        # if not self.is_evaluating:
        #     if self.config.domain_rand.push_robots:
        #         push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
        #         self.push_robot_counter[push_robot_env_ids] = 0
        #         self.push_robot_plot_counter[push_robot_env_ids] = 0
        #         self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
        #         self._push_robots(push_robot_env_ids)


        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)

    def _post_compute_observations_callback(self):
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.simulator.dof_pos[:]
        self.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]
        self.last_left_ee_vel[:] = self.simulator._rigid_body_vel[:, self.left_hand_link_index, :].view(self.num_envs, -1)
        self.last_left_ee_ang_vel[:] = self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, :].view(self.num_envs, -1)
        self.last_right_ee_vel[:] = self.simulator._rigid_body_vel[:, self.left_hand_link_index, :].view(self.num_envs, -1)
        self.last_right_ee_ang_vel[:] = self.simulator._rigid_body_ang_vel[:, self.right_hand_link_index, :].view(self.num_envs, -1)


    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()

        self.reset_buf |= self.time_out_buf

    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            self.reset_buf |= torch.any(torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)
        if self.config.termination.terminate_by_low_height:
            self.reset_buf |= torch.any(self.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)
            
            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                self.reset_buf |= out_of_dof_pos_limits > 0.
        
        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)
            
            

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                self.reset_buf |= out_of_dof_vel_limits > 0.
        
        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                self.reset_buf |= out_of_torque_limits > 0.


    def _update_timeout_buf(self):
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()


    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_tasks_callback(self, env_ids):
        self._episodic_domain_randomization(env_ids)
        if self.use_reward_penalty_curriculum:
            self._update_reward_penalty_curriculum()

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.base_lin_vel[env_ids] = target_buf["base_lin_vel"].to(self.base_lin_vel.dtype)
            self.base_ang_vel[env_ids] = target_buf["base_ang_vel"].to(self.base_ang_vel.dtype)
            self.projected_gravity[env_ids] = target_buf["projected_gravity"].to(self.projected_gravity.dtype)
            self.torques[env_ids] = target_buf["torques"].to(self.torques.dtype)
            self.actions[env_ids] = target_buf["actions"].to(self.actions.dtype)
            self.last_actions[env_ids] = target_buf["last_actions"].to(self.last_actions.dtype)
            self.last_dof_pos[env_ids] = target_buf["last_dof_pos"].to(self.last_dof_pos.dtype)
            self.last_dof_vel[env_ids] = target_buf["last_dof_vel"].to(self.last_dof_vel.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self.feet_air_time[env_ids] = target_buf["feet_air_time"].to(self.feet_air_time.dtype)
            self.last_contacts[env_ids] = target_buf["last_contacts"].to(self.last_contacts.dtype)
            self.last_contacts_filt[env_ids] = target_buf["last_contacts_filt"].to(self.last_contacts_filt.dtype)
            self.feet_air_max_height[env_ids] = target_buf["feet_air_max_height"].to(self.feet_air_max_height.dtype)
        else:
            self.actions[env_ids] = 0.
            self.last_actions[env_ids] = 0.
            self.actions_after_delay[env_ids] = 0.
            self.last_dof_pos[env_ids] = 0.
            self.last_dof_vel[env_ids] = 0.
            self.feet_air_time[env_ids] = 0.
            self.episode_length_buf[env_ids] = 0
            # self.reset_buf[env_ids] = 0
            # self.time_out_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._update_average_episode_length(env_ids)
            
            # for key in self.config.obs.obs_dict.keys():
            #     self.obs_buf_dict[key][env_ids] = torch.zeros((len(env_ids), self.dim_obs[key]*self.history_length[key]),
            #                                                   dtype=torch.float, device=self.device, requires_grad=False)

    def get_mppi_buffers(self, env_ids):
        """ Get buffers for MPPI
        MPPI algo need to store the buffers to replicate environments
        """
        return {
            "dof_pos": copy.deepcopy(self.simulator.dof_pos[env_ids]),
            "dof_vel": copy.deepcopy(self.simulator.dof_vel[env_ids]),
            "base_quat": copy.deepcopy(self.base_quat[env_ids]),
            "base_lin_vel": copy.deepcopy(self.base_lin_vel[env_ids]),
            "base_ang_vel": copy.deepcopy(self.base_ang_vel[env_ids]),
            "projected_gravity": copy.deepcopy(self.projected_gravity[env_ids]),
            "torques": copy.deepcopy(self.torques[env_ids]),
            "actions": copy.deepcopy(self.actions[env_ids]),
            "last_actions": copy.deepcopy(self.last_actions[env_ids]),
            "last_dof_pos": copy.deepcopy(self.last_dof_pos[env_ids]),
            "last_dof_vel": copy.deepcopy(self.last_dof_vel[env_ids]),
            "episode_length_buf": copy.deepcopy(self.episode_length_buf[env_ids]),
            "reset_buf": copy.deepcopy(self.reset_buf[env_ids]),
            "time_out_buf": copy.deepcopy(self.time_out_buf[env_ids]),
            "feet_air_time": copy.deepcopy(self.feet_air_time[env_ids]),
            "last_contacts": copy.deepcopy(self.last_contacts[env_ids]),
            "last_contacts_filt": copy.deepcopy(self.last_contacts_filt[env_ids]),
            "feet_air_max_height": copy.deepcopy(self.feet_air_max_height[env_ids]),
        }

    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
        self.log_dict["average_episode_length"] = self.average_episode_length

    def _compute_observations(self):
        """ Computes observations
        """
        self.obs_buf_dict_raw = {}
        
        # compute Algo observations
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = dict()
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.config.obs.obs_scales, self.config.obs.noise_scales)
        
        self._post_config_observation_callback()

    def _post_config_observation_callback(self):
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            obs_keys = sorted(obs_config)
            current_obs_buf = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            self.obs_buf_dict[obs_key] = self.obs_buf_dict[obs_key].to(self.device)
            self.obs_buf_dict[obs_key] = torch.cat((self.obs_buf_dict[obs_key][:, self.dim_obs[obs_key]:(self.dim_obs[obs_key]*self.history_length[obs_key])], 
                                                    current_obs_buf), dim=-1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
        else:
            return torques

    def _create_terrain(self):
        super()._create_terrain()

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self._refresh_sim_tensors()
        
        # draw push robot
        draw_env_ids = (self.push_robot_plot_counter < 10).nonzero(as_tuple=False).flatten()
        not_draw_env_ids = (self.push_robot_plot_counter >= 10).nonzero(as_tuple=False).flatten()
        self.record_push_robot_vel_buf[not_draw_env_ids] *=0
        self.push_robot_plot_counter[not_draw_env_ids] = 0
        
        for env_id in draw_env_ids:
            push_vel = self.record_push_robot_vel_buf[env_id]
            push_vel = torch.cat([push_vel, torch.zeros(1, device=self.device)])
            push_pos = self.simulator.robot_root_states[env_id, :3]
            push_vel_list = [push_vel]
            push_pos_list = [push_pos]
            push_mag_list = [1]
            push_color_schems = [(0.851, 0.144, 0.07)]
            push_line_widths = [0.03]
            for push_vel, push_pos, push_mag, push_color, push_line_width in zip(push_vel_list, push_pos_list, push_mag_list, push_color_schems, push_line_widths):
                for _ in range(200):
                    gymutil.draw_line(Point(push_pos +torch.rand(3, device=self.device) * push_line_width),
                                        Point(push_pos + push_vel * push_mag),
                                        Point(push_color),
                                        self.gym, self.viewer, self.envs[env_id])


    ################ Curriculum #################

    def _update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids], dtype=torch.float)
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)


    #------------ reward functions----------------
    ########################### PENALTY REWARDS ###########################

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_termination_lower_body(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_termination_upper_body(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_penalty_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1)
    
    def _reward_penalty_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_penalty_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_penalty_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.simulator.robot_root_states[:, 7:13]) / self.dt), dim=-1)
    
    def _reward_penalty_root_acc_upper_body(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.simulator.robot_root_states[:, 7:13]) / self.dt), dim=-1)
    
    def _reward_penalty_root_acc_lower_body(self):
        # Penalize root accelerations
        root_acc_linear_xy = (self.simulator.robot_root_states[:, 7:9] - self.last_root_vel[:, 0:2]) / self.dt
        root_acc_angular_xy = (self.simulator.robot_root_states[:, 10:12] - self.last_root_vel[:, 3:5]) / self.dt
        return torch.sum(torch.square(root_acc_linear_xy) + torch.square(root_acc_angular_xy), dim=-1)
        return torch.sum(torch.square((self.last_root_vel - self.simulator.robot_root_states[:, 7:13]) / self.dt), dim=-1)

    ######################## LIMITS REWARDS #########################

    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.rewards.reward_limit.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.rewards.reward_limit.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_penalty_slippage(self):
        # assert self.simulator._rigid_body_vel.shape[1] == 20
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)

    def _reward_feet_max_height_for_this_air(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)
        self.last_contacts = contact
        self.last_contacts_filt = contact_filt
        self.feet_air_max_height = torch.max(self.feet_air_max_height, self.simulator._rigid_body_pos[:, self.feet_indices, 2])

        rew_feet_max_height = torch.sum((torch.clamp_min(self.config.rewards.desired_feet_max_height_for_this_air - self.feet_air_max_height, 0)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_max_height *= ~contact_filt
        return rew_feet_max_height
    


    def _episodic_domain_randomization(self, env_ids):
        """ Update scale of Kp, Kd, rfi lim"""
        if len(env_ids) == 0:
            return
        if self.config.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.config.domain_rand.kp_range[0], self.config.domain_rand.kp_range[1], (len(env_ids), self.num_dofs), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.config.domain_rand.kd_range[0], self.config.domain_rand.kd_range[1], (len(env_ids), self.num_dofs), device=self.device)
    
        if self.config.domain_rand.randomize_rfi_lim:
            self._rfi_lim_scale[env_ids] = torch_rand_float(self.config.domain_rand.rfi_lim_range[0], self.config.domain_rand.rfi_lim_range[1], (len(env_ids), self.num_dofs), device=self.device)

        if self.config.domain_rand.randomize_ctrl_delay:            
            # self.action_queue[env_ids] = 0.delay:
            self.action_queue[env_ids] *= 0.
            # self.action_queue[env_ids] = 0.
            self.action_delay_idx[env_ids] = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                            self.config.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)


    def _push_robots(self, env_ids):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel = self.config.domain_rand.max_push_vel_xy
        self.push_robot_vel_buf[env_ids] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2), device=str(self.device))  # lin vel x/y
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.simulator.all_root_states))


    ############ TERRAIN AND COMMANDS

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device))
            
            self.simulator.dof_vel[env_ids] = 0.

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                     gymtorch.unwrap_tensor(self.simulator.dof_state),
        #                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids, target_root_states=None):
        """ Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]

        else:
            # base position
            if self.custom_origins:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
                self.simulator.robot_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=str(self.device)) # xy position within 1m of the center
            else:
                self.simulator.robot_root_states[env_ids] = self.base_init_state
                self.simulator.robot_root_states[env_ids, :3] += self.env_origins[env_ids]
            # base velocities
            
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=str(self.device)) # [7:10]: lin vel, [10:13]: ang vel


    def _plot_domain_rand_params(self):
        raise NotImplementedError

    ######################### Observations #########################
    def _get_obs_base_pos_z(self,):
        return self.simulator.robot_root_states[:, 2:3]
    
    def _get_obs_feet_contact_force(self,):
        return self.simulator.contact_forces[:, self.feet_indices, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        return self.base_lin_vel
    
    def _get_obs_base_ang_vel(self,):
        return self.base_ang_vel
    
    def _get_obs_projected_gravity(self,):
        return self.projected_gravity
    
    def _get_obs_dof_pos(self,):
        return self.simulator.dof_pos - self.default_dof_pos
    
    def _get_obs_dof_vel(self,):
        return self.simulator.dof_vel
    
    def _get_obs_actions(self,):
        return self.actions