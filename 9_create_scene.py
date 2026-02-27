"""
이 스크립트에서는 scene.InteractiveScene 클래스에 대해 소개한다.
이는 prim을 소환하고 그들을 시뮬레이션에서 다루기 위한 편리한 인터페이스를 제공한다.

High-level에서 interactive scene은 scene의 entity들의 모음이다.
각 entity는 non-interactive prime (e.g. ground plane, light source) 일 수도 있고, 
interactive prim (e.g. articulation, rigid object) 혹은 sensor(e.g. camera, lidar) 일 수 있다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from isaaclab_assets import CARTPOLE_CFG

# entitiy들의 집합으로 scene이 이루어지는데, 이들은 scene.InteractiveSceneCfg 로부터 구체화됨
# 여기서 사용한 변수 명은 scene.InteractiveScene object로부터 entity에 대해 접근하기 위한 key로 사용된다. (scene["cartpole"])

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    
    # ground plane - non-interactive prims
    ground = AssetBaseCfg(prim_path="/World/defaultGridPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # lights - non-interactive prims
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # articulation - interactive prim
    # cartpole의 경우 relative path를 사용하여 구체화 되는데
    # relative path는 ENV_REGEX_NS 변수를 사용하여 구체화되고, 이는 scene이 생성되면서 환경 이름으로 교체되는
    # special 변수이다. [ENV_REGEX_NS -> /World/envs/env_{i}]
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability
    robot = scene["cartpole"]
    
    sim_dt = sim.get_physics_dt()
    count = 0
    
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset the scene entities
            # root state
            # we offset the root state by the origin
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffer
            scene.reset()
            print("[INFO]: Resetting robot state...")
            
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        
        # -- write data to simi
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        
        # Increment counter
        count += 1
        
        # Update buffers
        scene.update(sim_dt)
        
def main():
    """Main function"""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene)
    
if __name__ == "__main__":
    main()
    simulation_app.close()