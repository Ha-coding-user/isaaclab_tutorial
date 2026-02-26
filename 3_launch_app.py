"""
This script demonstrates how to run IsaacSim via the AppLauncher
"""

import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")

# SimulationApp arguments
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Default to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Default to 720"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils

def design_scene():
    """Design the scene by spawning ground plane, light, objects and meshes from usd files."""
    
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    
    # spawn a cuboid
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        # PreviewSurfaceCfg는 prim에 기본 PBR 재질(색/거칠기/금속성/투명도 등) 을 간단히 입히는 용도
        # diffuse_color / emissive_color / roughness / metallic / opacity
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
    )
    # spawn cuboid, altering translation on the z-axis to scale to its size
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))
    
def main():
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    
    # Design scene by adding assets to it
    design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Now ready
    print("[INFO]: Setup complete...")
    
    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()
        
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app 
    simulation_app.close()