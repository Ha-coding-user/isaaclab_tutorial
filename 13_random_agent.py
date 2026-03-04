"""
이전 튜토리얼에서 custom cartpole 환경을 어떻게 만들었는지 학습했다.
우리는 수작업으로 environment 클래스와 그의 configuration 클래스를 import 하여 환경의 instance를 만들었다.

직관적으로 이 방법은 아주 큰 환경으로 확장이 어렵다.
이 튵리얼에서 우리는 gymnasium 레지스트리에 환경을 등록하기 위한 gymnasium.register() 함수를 어떻게 쓰는지 보인다.
이는 우리가 gymnasium.make() 함수들을 통해 환경을 만들도록 해준다.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

"""
envs.ManagerBasedRLEnv는 gymnasium.Env를 상속하기에, RL 라이브러리들이 기대하는 표준 API를 맞춤.
그런데 전통적 Gym env는 보통 환경 instance 1개 = state 1개임.
반면 ManagerBasedRLEnv는 한 instance 안에 N개의 sub-environment를 들고 있고, 동시에 step 하기에 반환되는 데이터가 전부 배치:
    - obs: (num_envs, obs_dim)
    - action: (num_envs, action_dim)
    - reward: (num_envs,)
    - terminated/truncated: (num_envs,)
    
envs.DirectRLEnv도 gymnasium.Env를 상속해서 Gym API를 맞춤.
차이는 Manager들을 통해 terms를 조립하기 보다는, 보상/관측/종료/리셋 로직 등을 env 클래스 안에서 직접 코드로 구현하는 스타일

DirectMARLEnv는 Multi-Agent RL용으로 Gymnasium을 상속하지 않을 수 있지만, Isaac Lab의 registry/팩토리 시스템에 맞춰
gymnasium.make()처럼 똑같이 등록하고 생성할 수 있도록 해둠
"""

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            env.step(actions)
            
    # close the simulator
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()