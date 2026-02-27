"""
Isaac Lab에서 manager 기반의 환경들은 envs.ManagerBasedEnv와 envs.ManagerBasedRLEnv 클래스들에 의해 구현됨.
2개의 클래스들은 매우 유사하지만, envs.ManagerBasedRLEnv는 강화학습 작업에서 유용하며 보상, 종료, 커리큘럼 그리고
명령 생성을 포함함.
envs.ManagerBasedEnv 클래스는 전통적인 robot control에서 유용하고 보상과 termination을 포함하지 않는다.

이 스크립트에서는 envs.ManagerBasedEnv를 보고 그에 대응하는 configure class 인 envs.ManagerBasedEnvCfg를 확인한다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on creating cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg

# 이전에는 cartpole에 action을 입력할 때 assets.Articulation.set_joint_effort_target() 을 사용했었음.
# 여기서는 managers.ActionManager를 사용함
# action manager는 여러개의 managers.ActionTerm으로 구성되어 있음.
# 각 action term은 환경의 특정 측면에 대한 제어를 담당함 (e.g., 로봇 팔의 경우 joint, gripper를 제어하기 위한 두 가지 term)
@configclass
class ActionsCfg:
    """Action specifications for the environments."""
    
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)

# scene이 환경의 상태에 대해 정의하면, observations는 agent에 의해 관측될 수 있는 상태들을 정의함
# 이러한 observation들은 어떤 행동을 할지 결정을 만드는데 사용됨.
# Isaac Lab에서 observation들은 managers.ObservationManager 클래스에 의해 계산됨

# 여기서 "policy"라고 부르는 하나의 observation 그룹만 정의함
# managers.ObservationGroupCfg 클래스는 다른 observation term들을 수집하고 그룹에 대해 공통 특징들을 정의하는 것을 도와줌
# 개별적인 term들은 managers.ObservatgionTermCfg 클래스로부터 정의됨
@configclass
class ObservationCfg:
    """Observation specifications for the environment."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        joint_pos_vel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
# managers.EventManager 클래스는 시뮬레이션 상태 변화에 대응하는 이벤트를 담당함
# 이는 scene을 resetting (혹은 randomizing), 물리적 특징들(mass, friction 등) + 시각적 특징들(색, 텍스쳐..) 를 랜덤화
# 이들 각각은 managers.EventTermCfg 클래스를 통해 구체화되고, 이는 이벤트를 수행하는 함수 혹은 호출 가능 클래스를 지정하는
# managers.EventTermCfg.func를 인수로 받음

# 모드는 Event Term이 적용되어야 하는 시점을 지정함.
# 이를 위해서 ManagerBasedEnv 클래스를 수정해야 함.
# 그러나 기본적으로 Isaac Lab은 세 가지 일반적으로 사용되는 모드를 제공함:
#   - startup: 환경이 시작될 때 한 번 place를 받음
#   - reset: 환경이 종료 되거나 reset 될때 발생
#   - interval: 주기적으로 실행되는 event
@configclass
class EventCfg:
    """Configuration for events."""
    
    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add"
        }
    )
    
    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        }
    )
    
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        }
    )
    
# 이 클래스를 통해 모두 묶음
# envs.ManagerBasedEnvCfg 클래스릍 통해 환경의 configuration을 정의함
# 이 클래스는 scene, action, observation 그리고 event configuration을 받음
@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""
    
    # scene setting
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    
    # Basic setting
    observations = ObservationCfg()
    actions = ActionsCfg()
    events = EventCfg()
    
    # configuration이 초기화 된 후 호출되는 함수
    def __post_init__(self):
        """Post initialization"""
        # viewer setting
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        
        # step setting
        self.decimation = 4 # env step every 4 sim steps: 200Hz / 4 = 50Hz
        
        # simulation setting
        self.sim.dt = 0.005
        
def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)
    
    # simulate physics
    count = 0
    while simulation_app.is_running():
        # torch.interface_mode()를 사용하는 이유는 환경이 내부적으로 Pytorch 연산을 사용하기 때문이며
        # 시뮬레이션이 Pytorch 자동 미분 엔진의 오버헤드로 인해 느려지지 않도록 하고
        # 시뮬레이션 연산에 대해 기울기가 계산되지 않도록 하기 위함
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1
            
    # close the environment
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()