"""
옷이나, 액체 등과 같은 deformable objects 에 대해 PhysX에서 soft한 body로 정의
rigid body와 달리 soft body들은 외부 힘이나 충돌에 의해 변형됨.

soft body는 두 개의 사면체 메쉬로 구성됨 - simulation mesh & collision mesh
    - simulation mesh: soft body의 변형을 시뮬레이션 하기 위해 사용됨
    - collision mesh: scene에서 다른 물체들과의 충돌을 감지하는데 사용됨
    
이 스크립트에서는 어떻게 시뮬레이션에서 deformable object가 상호작용하는지 보여줌
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on interacting with a deformable objects.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defulatGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    
    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)
        
    # Deformable Object
    cfg = DeformableObjectCfg(
        prim_path="/World/Origin.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True
    )
    cube_object=DeformableObject(cfg=cfg)
    
    # return the scene information
    scene_entities = {"cube_object": cube_object}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject], origins: torch.Tensor):
    """Runs the simulation loop"""
    cube_object = entities["cube_object"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Nodal kinematic targets of the deformable bodies
    # nodal kinematic은 deformable body의 각 노드를 물리적으로 풀어 움직이게 할지(dynamic)
    # 내가 지정한 목표 위치로 강제로 끌고 갈지(kinematic)를 나타내는 개념
    nodal_kinematic_target = cube_object.data.nodal_kinematic_target.clone()    # 값 체크해볼 필요 있음
    
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            
            # reset the nodal state of the object
            nodal_state = cube_object.data.default_nodal_state_w.clone()
            
            # apply random pose to object
            pos_w = torch.rand(cube_object.num_instances, 3, device=sim.device) * 0.1 + origins
            quat_w = math_utils.random_orientation(cube_object.num_instances, device=sim.device)
            nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)
            
            # Write the nodal state to the kinematic target and free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            
            # reset buffers
            cube_object.reset()
            
            print("------------------------------------")
            print("[INFO]: Resetting object state...")
            
            
        # update the kinematic target for cubes at index 0 and 3
        # we slightly move the cube in the z-direction by picking the vertex at index 0
        nodal_kinematic_target[[0, 3], 0, 2] += 0.001
        
        # set vertex at index 0 to be kinematically constrained
        # 0: constrained, 1: free
        nodal_kinematic_target[[0, 3], 0, 3] = 0.0
        
        # write kinematic target to simulation
        cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
        
        # write internal data to simulation
        cube_object.write_data_to_sim()
        
        sim.step()
        
        sim_time += sim_dt
        count += 1
        
        cube_object.update(sim_dt)
        
        if count % 50 == 0:
            print(f"Root position (in world): {cube_object.data.root_pos_w[:, :3]}")
            
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])
    
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    sim.reset()
    
    print("[INFO]: Setup complete...")
    
    run_simulator(sim, scene_entities, scene_origins)
    
if __name__ == "__main__":
    main()
    simulation_app.close()