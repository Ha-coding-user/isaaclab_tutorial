"""
이 스크립트에선 어떻게 강화 학습을 위한 manager 기반 task environment를 만드는지 확인한다.

base environment는 agent가 명령을 환경으로 보내고 환경으로부터 observation을 받는 sense-act 환경으로 설계되었다.
이는 전통직인 motion plannint & controls와 같은 많은 application에 충분했다.
하지만 많은 application들은 에이전트의 학습 목표로 작용하는 task-specification을 요구한다.
이를 위해 base environment를 task specification으로 확장하는 envs.ManagerBasedRLEnv 클래스를 사용한다.

Isaac Lab에서 task 환경을 위해 configuration envs.ManagerBasedRLEnvCfg 를 구현하길 권장한다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environments.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from _11_cartpole_env_cfg import CartpoleEnvCfg

def main():
    """Main function."""

    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # setup RL envrionment
    env = ManagerBasedRLEnv(env_cfg)
    
    # simulate physics
    count = 0
    while simulation_app.is_running():
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
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            
            # update counter
            count += 1
            
    # close the envrionment
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()