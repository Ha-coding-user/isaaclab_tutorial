import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Robot은 관절을 가진 관절 구조물이기 때문에, 이를 조절하기 위한 ArticulationCfg를 정의하여 로봇을 묘사
JETBOT_CONFIG = ArticulationCfg(
    # Isaac Lab sim_utils는 USD asset의 경로를 처리하는 USDFileCfg를 제공하고, 이로부터 SpawnerCfg를  생성함
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    # actuators는 dictionary 형태이고, 로봇의 어떤 부분을 조절할지 정의
    # ImplicitActuatorCfg는 내장 joint drive를 이용하도록 하며, 시뮬레이터가 한 스텝 안에서 안정적으로 목표를 추종하도록
    # 내부적으로 처리하는 형태라, 비교적 큰 dt에서도 안정적임
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)}
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        # 물리적 특성에 대한 구성요소 추가 (Jetbot과 다른 부분)
        # RigidBodyPropertiesCfg를 통해 시뮬레이션 내에서 "물리적 객체"로서의 동작과 관련된
        # 생성되는 로봇의 바디 링크 속성을 지정
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # articulation_props는 관절을 시간에 따라 단계적으로 이동시키는데 사용되는
        # 솔버와 관련된 속성을 제어하므로, ArticulationRootPropertiesCfg가 구성되기를 기대
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    # 로봇의 초기상태 (Jetbot과 다른 부분)
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0
        },
        pos=(0.25, -0.25, 0.0), # environment를 기준으로 한 pos 값
    ),
    # 액션을 넣으면, 이 액션이 어떤 조인트에 어떻게 적용될지를 actuator가 결정
    actuators={
        "font_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0
        )
    }
)

class NewRobotSceneCfg(InteractiveSceneCfg):
    """Designs the scene"""
    
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins
            
            # copy the default root state to the sim for the jetbot's orientation and velocity
            # 일반적으로 [num_envs, ...]
            # [:3] = position (x, y, z)
            # [3:7] = orientation quaternion(x, y, z, w)
            # [7:] = linear/angular velocity (보통 6개: vx,vy,vz / wx,wy,wz)
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
    
            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone()
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")
            
        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])
            
        scene["Jetbot"].set_joint_velocity_target(action)
        
        # wave
        wave_action = scene["Dofbot"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Dofbot"].set_joint_position_target(wave_action)
        
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
def main():
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_cfg = NewRobotSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # play the simulator
    sim.reset()
    
    # now ready
    print("[INFO]: Setup complete...")
    
    # run the simulator
    run_simulator(sim, scene)
    
if __name__ == "__main__":
    main()
    simulation_app.close()