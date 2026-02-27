import math

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
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # cartpole
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    
#####################
# MDP Setting
#####################

@configclass
class ActionCfg:
    """Action specification for the MDP."""
    
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    
@configclass
class ObservationCfg:
    """Observation specification for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class EventCfg:
    """configuration for events."""
    
    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5)
        }
    )
    
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)
        }
    )
    
# managers.RewardManager는 agnet에 대한 reward term을 계산하기 위해 사용됨.
# 다른 magager들과 유사하게 이 term들은 managers.RewardTermCfg 클래스를 사용하여 configure 됨
# 이 Task에서는 다음 reward term을 사용함
#   - Alive Rewarding: agent가 최대한 오래동안 alive할 수 있도록 함
#   - Terminating Reward: terminating하면 penalty 부여
#   - Pole Angle Reward: agent가 upright position을 유지하도록 함
#   - Cart Velocity Reward: agent가 cart 속도를 최대한 작게 유지하도록 함
#   - Pole Velocity Reward: agent가 pole의 속도를 최대한 작게 유지하도록 함
@configclass
class RewardCfg:
    """Reward terms for the MDP."""
    
    # (1) constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
        
    )
    #(4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])}
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])}
    )
    
@configclass
class TerminationCfg:
    """Termination terms for the MDP."""
    
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)}
    )
    
#####################
# Environment Configuration
#####################

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""
    
    # Scene setting
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    # Basic setting
    observations: ObservationCfg = ObservationCfg()
    actions: ActionCfg = ActionCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards : RewardCfg = RewardCfg()
    terminations : TerminationCfg = TerminationCfg()
    
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization"""
        # general setting
        self.decimation = 2
        self.episode_length_s = 5
        
        # viewer setting
        self.viewer.eye = (8.0, 0.0, 5.0)
        
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation