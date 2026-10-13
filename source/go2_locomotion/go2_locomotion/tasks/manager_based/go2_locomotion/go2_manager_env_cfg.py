"""Manager based Go2 quadruped and handstand locomotion environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2, update_period=0.005
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state.pos = (0.0, 0.0, 0.42)
        self.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0, ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5,
        }
        self.robot.actuators["base_legs"].stiffness = 30.0
        self.robot.actuators["base_legs"].damping = 0.75


@configclass
class CommandsCfg:
    base_velocity = mdp.Go2CommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=False)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=mdp.JOINT_NAMES, preserve_order=True,
        scale=0.25, use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        history = ObsTerm(func=mdp.Go2History, params={"critic": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        history = ObsTerm(func=mdp.Go2History, params={"critic": True})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    tracking_lin_vel = RewTerm(func=mdp.tracking_lin_vel, weight=1.5)
    tracking_ang_vel = RewTerm(func=mdp.tracking_ang_vel, weight=1.0)
    gait_quad = RewTerm(func=mdp.gait, weight=0.5, params={"hand": False})
    gait_hand = RewTerm(func=mdp.gait, weight=0.5, params={"hand": True})
    height_quad = RewTerm(func=mdp.base_height, weight=1.0, params={"hand": False})
    height_hand = RewTerm(func=mdp.base_height, weight=1.0, params={"hand": True})
    orientation_quad = RewTerm(func=mdp.body_orientation, weight=1.0, params={"hand": False})
    orientation_hand = RewTerm(func=mdp.body_orientation, weight=1.0, params={"hand": True})
    clearance_quad = RewTerm(func=mdp.foot_clearance, weight=0.2, params={"hand": False})
    clearance_hand = RewTerm(func=mdp.foot_clearance, weight=0.2, params={"hand": True})
    rear_contact = RewTerm(func=mdp.rear_contact, weight=-2.0)
    rear_joint = RewTerm(func=mdp.rear_joint, weight=-0.1)
    contact_velocity = RewTerm(func=mdp.contact_velocity, weight=-0.1)
    hip_position = RewTerm(func=mdp.hip_position, weight=-1.0)
    collision = RewTerm(func=mdp.collision, weight=-1.5)
    torques = RewTerm(func=mdp.torques, weight=-1.0e-4)
    joint_velocity = RewTerm(func=mdp.joint_velocity, weight=-1.0e-4)
    joint_acceleration = RewTerm(func=mdp.joint_acceleration, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate, weight=-1.0e-3)
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_contact = DoneTerm(func=mdp.bad_contact, params={"threshold": 1.0})


@configclass
class EventsCfg:
    material = EventTerm(func=mdp.randomize_go2_material, mode="startup")
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-1.0, 1.0), "operation": "add"},
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}},
    )
    joint_physics = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.005, 0.05),
                "armature_distribution_params": (0.01, 0.05), "operation": "abs"},
    )
    joint_damping = EventTerm(func=mdp.randomize_joint_damping, mode="startup")
    cache_privileged = EventTerm(func=mdp.cache_privileged_params, mode="startup")
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
                "velocity_range": {axis: (-0.1, 0.1) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}},
    )
    reset_hip = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                "position_range": (-0.2, 0.2), "velocity_range": (-0.2, 0.2)},
    )
    reset_thigh = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
                "position_range": (-0.4, 0.4), "velocity_range": (-0.2, 0.2)},
    )
    reset_calf = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
                "position_range": (-0.4, 0.4), "velocity_range": (-0.2, 0.2)},
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(8.0, 8.0),
        is_global_time=True,
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                                   "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}},
    )


@configclass
class Go2Quad2HandEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
