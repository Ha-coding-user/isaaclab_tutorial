"""
여기선 end-effector에 붙어있는 surface gripper가 있는 articulated robot이 어떻게 상호작용하는지 확인.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets import PICK_AND_PLACE_CFG

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defualtGridPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, -.75))
    cfg.func("/World/Light", cfg)
    
    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[2.75, 0.0, 0.0], [-2.75, 0.0, 0.0]]
    
    # Origin 1
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    sim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    
    # Articulation: First we defie the robot config
    # pick-and-place robot은 단순한 3축 로봇으로, x, y축을 따라 이동 + z축을 따라 상하로 움직임
    # robot의 end-effector는 surface-gripper가 장착되어 있음
    pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
    pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
    pick_and_place_robot = Articulation(cfg=pick_and_place_robot_cfg)
    
    # Surface Gripper: Next we define the surface gripper config
    surface_gripper_cfg = SurfaceGripperCfg()
    # We need to tell the View which prim to use for the surface gripper
    surface_gripper_cfg.prim_path = "/World/Origin.*/Robot/picker_head/SurfaceGripper"
    
    # We can then set different parameters for the surface gripper, note that if these params are not set,
    # the View will try to read them from the prim.
    surface_gripper_cfg.max_grip_distance = 0.1     # Maximum distance at which the gripper can grasp an object
    surface_gripper_cfg.shear_force_limit = 500.0   # Force limit in the direction perpendicular direction
    surface_gripper_cfg.coaxial_force_limit = 500.0 # Force limit in the direction of the gripper's axis
    surface_gripper_cfg.retry_interval = 0.1    # Time the gripper will stay in a grasping state
    
    # Spawn the surface gripper
    surface_gripper = SurfaceGripper(cfg=surface_gripper_cfg)
    
    # return the scene information
    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, origins

def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]
    
    sim_dt = sim.get_physics_dt()
    count = 0
    
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0   # reset
            
            # reset the scene entities
            # root state
            # offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robot will be spawned at the (0, 0, 0) of the simulation
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
            
            # Opens the gripper and makes sure the gripper is in the open state
            surface_gripper.reset()
            print("[INFO]: Resetting grippr state...")
            
        # Sample a random command between -1 and 1.
        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
        
        # The gripper behavior is as follows:
        # -1 < commadnd < -0.3 --> Gripper is opening
        # -0.3 < command < 0.3 --> Gripper is Idle
        # 0.3 < command < 1 --> Gripper is Closing
        print(f"[INFO]: Gripper commands: {gripper_commands}")
        mapped_commands = [
            "Opening" if command < -0.3 else "Closing" if command > 0.3 else "Idle" for command in gripper_commands
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")
        
        # Set the gripper command
        surface_gripper.set_grippers_command(gripper_commands)
        surface_gripper.write_data_to_sim()
        
        sim.step()
        count += 1
        
        surface_gripper.update(sim_dt)
        surface_gripper_state = surface_gripper.state
        
        print(f"[INFO]: Gripper state: {surface_gripper_state}")
        mapped_commands = [
            "Open" if state == -1 else "Closing" if state == 0 else "Closed" for state in surface_gripper_state
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")
        
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
    
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)
    
if __name__ == "__main__":
    main()
    simulation_app.close()