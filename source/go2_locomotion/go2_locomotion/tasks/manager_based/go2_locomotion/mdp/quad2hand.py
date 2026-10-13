"""Manager terms for the Go2 quadruped and handstand task."""

from __future__ import annotations

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat


JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
FOOT_NAMES = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
BAD_CONTACT_NAMES = ("base", "Head_upper", "Head_lower", "FL_hip", "FR_hip", "RL_hip", "RR_hip")


class Go2Command(CommandTerm):
    """Samples mode-specific velocity commands and owns the gait/contact phase state."""

    def __init__(self, cfg: Go2CommandCfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._periods = torch.tensor([0.5, 0.4], device=self.device)
        self._offsets = torch.tensor([[0.0, 0.5, 0.5, 0.0], [0.0, 0.5, 0.0, 0.0]], device=self.device)
        robot = env.scene["robot"]
        sensor = env.scene["contact_sensor"]
        self.joint_ids = torch.tensor([robot.find_joints(name)[0][0] for name in JOINT_NAMES], device=self.device)
        self.foot_ids = torch.tensor([sensor.find_bodies(name)[0][0] for name in FOOT_NAMES], device=self.device)
        self.base_id = robot.find_bodies("base")[0][0]
        self.bad_contact_ids = torch.tensor(
            sorted({i for name in BAD_CONTACT_NAMES for i in sensor.find_bodies(name)[0]}), device=self.device
        )
        self.collision_ids = torch.tensor(
            sorted({i for part in ("thigh", "calf") for i in sensor.find_bodies(f".*_{part}")[0]}), device=self.device
        )
        self.base_mass = robot.root_physx_view.get_masses()[:, self.base_id:self.base_id + 1].to(self.device)
        self.friction = robot.root_physx_view.get_material_properties()[:, 0, :1].to(self.device)
        self._last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._filtered_contacts = self._last_contacts.clone()
        self._contact_step = -1

    @property
    def command(self):
        return self._command

    @property
    def leg_phase(self):
        period = self._periods[self.mode]
        phase = (self._env.episode_length_buf * self._env.step_dt) % period / period
        return (phase[:, None] + self._offsets[self.mode]) % 1.0

    @property
    def clock(self):
        phase = self.leg_phase
        return torch.cat((torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)), dim=-1)

    def contacts(self):
        # Multiple reward terms use this value within one environment step.
        if self._contact_step != self._env.common_step_counter:
            sensor = self._env.scene["contact_sensor"]
            current = torch.linalg.vector_norm(sensor.data.net_forces_w[:, self.foot_ids], dim=-1) > 1.0
            self._filtered_contacts = current | self._last_contacts
            self._last_contacts = current
            self._contact_step = self._env.common_step_counter
        return self._filtered_contacts

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_contacts[env_ids] = False
        self._filtered_contacts[env_ids] = False
        self._contact_step = -1
        return super().reset(env_ids)

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids):
        self.mode[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
        for mode, ranges in (
            (0, ((-1.0, 1.0), (-0.8, 0.8), (-1.0, 1.0))),
            (1, ((-0.4, 0.4), (0.0, 0.0), (-0.5, 0.5))),
        ):
            chosen = env_ids[self.mode[env_ids] == mode]
            for axis, (low, high) in enumerate(ranges):
                self._command[chosen, axis] = low + (high - low) * torch.rand(len(chosen), device=self.device)
        self._command[env_ids] *= self._command[env_ids].abs() > 0.1


@configclass
class Go2CommandCfg(CommandTermCfg):
    class_type: type = Go2Command


def task(env):
    return env.command_manager.get_term("base_velocity")


def _robot(env):
    return env.scene["robot"].data


def policy_frame(env):
    term = task(env)
    data = _robot(env)
    joints = term.joint_ids
    frame = torch.cat((
        data.root_ang_vel_b * 0.25,
        data.projected_gravity_b,
        term.command * torch.tensor([2.0, 2.0, 0.25], device=env.device),
        data.joint_pos[:, joints] - data.default_joint_pos[:, joints],
        data.joint_vel[:, joints] * 0.05,
        env.action_manager.action,
        term.clock,
        torch.nn.functional.one_hot(term.mode, num_classes=2).float(),
    ), dim=-1)
    noise = torch.zeros(55, device=env.device)
    noise[:3] = 0.2 * 0.25
    noise[3:6] = 0.05
    noise[9:21] = 0.01
    noise[21:33] = 1.5 * 0.05
    return frame + (2.0 * torch.rand_like(frame) - 1.0) * noise


def critic_frame(env):
    term = task(env)
    data = _robot(env)
    joints = term.joint_ids
    clean = torch.cat((
        data.root_ang_vel_b * 0.25,
        data.projected_gravity_b,
        term.command * torch.tensor([2.0, 2.0, 0.25], device=env.device),
        data.joint_pos[:, joints] - data.default_joint_pos[:, joints],
        data.joint_vel[:, joints] * 0.05,
        env.action_manager.action,
        term.clock,
        torch.nn.functional.one_hot(term.mode, num_classes=2).float(),
    ), dim=-1)
    actuator = env.scene["robot"].actuators["base_legs"]
    return torch.cat((
        clean, data.root_lin_vel_b * 2.0, term.friction, term.base_mass / 20.0,
        data.body_com_pos_b[:, term.base_id], data.joint_friction_coeff[:, joints[:1]],
        data.joint_damping[:, joints[:1]], data.joint_armature[:, joints[:1]],
        actuator.stiffness[:, joints] / 30.0, actuator.damping[:, joints] / 0.75,
        torch.ones(env.num_envs, 12, device=env.device),
        torch.zeros(env.num_envs, 12, device=env.device),
    ), dim=-1)


class Go2History(ManagerTermBase):
    """Current frame followed by four most recent frames, as expected by the old policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.history = None
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.last_step[env_ids] = -1
        if self.history is not None:
            self.history[env_ids] = 0.0

    def __call__(self, env, critic: bool = False):
        frame = critic_frame(env) if critic else policy_frame(env)
        if self.history is None:
            self.history = torch.zeros(env.num_envs, 4, frame.shape[1], device=env.device)
        result = torch.cat((frame, self.history.flatten(1)), dim=-1)
        changed = self.last_step != env.common_step_counter
        self.history[changed] = torch.cat((frame[changed, None], self.history[changed, :-1]), dim=1)
        fresh = changed & (env.episode_length_buf <= 1)
        self.history[fresh] = frame[fresh, None].expand(-1, 4, -1)
        self.last_step[changed] = env.common_step_counter
        return result.clamp(-100.0, 100.0)


def tracking_lin_vel(env):
    cmd = task(env)
    vel = _robot(env).root_lin_vel_w
    yaw = euler_xyz_from_quat(_robot(env).root_quat_w)[2]
    vx = vel[:, 0] * torch.cos(yaw) + vel[:, 1] * torch.sin(yaw)
    vy = -vel[:, 0] * torch.sin(yaw) + vel[:, 1] * torch.cos(yaw)
    error = (vx - cmd.command[:, 0]).square() + (cmd.mode == 0) * (vy - cmd.command[:, 1]).square()
    return torch.exp(-error / 0.25)


def tracking_ang_vel(env):
    return torch.exp(-(_robot(env).root_ang_vel_w[:, 2] - task(env).command[:, 2]).square() / 0.25)


def gait(env, hand: bool):
    cmd = task(env)
    contacts = cmd.contacts()
    stance = cmd.leg_phase < 0.6
    if hand:
        return (contacts[:, :2] == stance[:, :2]).float().mean(dim=1) * (cmd.mode == 1)
    return (contacts == stance).float().mean(dim=1) * (cmd.mode == 0)


def base_height(env, hand: bool):
    height = _robot(env).root_pos_w[:, 2]
    return torch.exp(-(height - (0.5 if hand else 0.36)).square() / (0.25 if hand else 0.04)) * (task(env).mode == int(hand))


def body_orientation(env, hand: bool):
    roll, pitch, _ = euler_xyz_from_quat(_robot(env).root_quat_w)
    target = 1.466 if hand else 0.0
    sigma = 0.25 if hand else 0.04
    return torch.exp(-(roll.square() + (pitch - target).square()) / sigma) * (task(env).mode == int(hand))


def foot_clearance(env, hand: bool):
    cmd = task(env)
    foot_ids = cmd.foot_ids[:2] if hand else cmd.foot_ids
    pos = _robot(env).body_pos_w[:, foot_ids]
    vel = _robot(env).body_lin_vel_w[:, foot_ids]
    error = (torch.linalg.vector_norm(vel[:, :, :2], dim=-1) * (pos[:, :, 2] - 0.072).square()).sum(dim=1)
    return torch.exp(-error / 0.01) * (cmd.mode == int(hand))


def rear_contact(env):
    cmd = task(env)
    return cmd.contacts()[:, 2:].float().sum(dim=1) * (cmd.mode == 1)


def rear_joint(env):
    cmd = task(env)
    data = _robot(env)
    ids = cmd.joint_ids[6:]
    return (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).square().sum(dim=1) * (cmd.mode == 1)


def contact_velocity(env):
    cmd = task(env)
    vel = _robot(env).body_lin_vel_w[:, cmd.foot_ids]
    return (vel.square().sum(dim=-1) * cmd.contacts()).sum(dim=1)


def hip_position(env):
    data = _robot(env)
    ids = task(env).joint_ids[[0, 3, 6, 9]]
    return (data.joint_pos[:, ids] - data.default_joint_pos[:, ids]).square().sum(dim=1)


def collision(env):
    ids = task(env).collision_ids
    forces = env.scene["contact_sensor"].data.net_forces_w[:, ids]
    return (torch.linalg.vector_norm(forces, dim=-1) > 0.1).float().sum(dim=1)


def torques(env):
    return _robot(env).applied_torque[:, task(env).joint_ids].square().sum(dim=1)


def joint_velocity(env):
    return _robot(env).joint_vel[:, task(env).joint_ids].square().sum(dim=1)


def joint_acceleration(env):
    return _robot(env).joint_acc[:, task(env).joint_ids].square().sum(dim=1)


def action_rate(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=1)


def bad_contact(env, threshold: float = 1.0):
    ids = task(env).bad_contact_ids
    forces = env.scene["contact_sensor"].data.net_forces_w_history[:, :, ids]
    return torch.any(torch.amax(torch.linalg.vector_norm(forces, dim=-1), dim=1) > threshold, dim=1)


def randomize_joint_damping(env, env_ids):
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    damping = robot.data.default_joint_damping[env_ids].clone()
    damping.uniform_(0.1, 0.5)
    robot.write_joint_damping_to_sim(damping, env_ids=env_ids)


def randomize_go2_material(env, env_ids):
    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    buckets = 0.2 + 1.05 * torch.rand(64, device="cpu")
    coefficients = buckets[torch.randint(0, 64, (len(env_ids),), device="cpu")]
    materials[env_ids, :, 0] = coefficients[:, None]
    materials[env_ids, :, 1] = coefficients[:, None]
    materials[env_ids, :, 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, env_ids)


def cache_privileged_params(env, env_ids):
    robot = env.scene["robot"]
    command = task(env)
    command.base_mass[:] = robot.root_physx_view.get_masses()[:, command.base_id:command.base_id + 1].to(env.device)
    command.friction[:] = robot.root_physx_view.get_material_properties()[:, 0, :1].to(env.device)
