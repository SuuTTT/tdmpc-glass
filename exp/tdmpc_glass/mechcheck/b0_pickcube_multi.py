"""
B0 task: PickCubeMulti-v1 — a LEARNABLE, multi-object, CONFIGURABLE-object-count
ManiSkill task for the compositional-OOD world-model gate.

Design: subclass the canonical learnable PickCube. The objective is unchanged
(grasp the RED target cube and move it to the goal). We additionally spawn
`num_cubes - 1` distractor cubes (other colors). All cubes' poses + tcp-relative
positions are included in the STATE obs, so the observation dimensionality and
object composition grow with the object-count knob `num_cubes`. The reward,
success condition, and target only depend on the red cube => PPO can learn it at
the train count, and we can probe value-decodability at held-out counts.

Object-count knob: env kwarg `num_cubes` (int >= 1). Set N via
  gym.make("PickCubeMulti-v1", num_cubes=N, num_envs=..., obs_mode="state")
"""
import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

_DISTRACTOR_COLORS = [
    [0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1],
    [0, 1, 1, 1], [1, 0.5, 0, 1], [0.5, 0.5, 0.5, 1], [0.3, 0.1, 0.6, 1],
]


@register_env("PickCubeMulti-v1", max_episode_steps=50)
class PickCubeMultiEnv(PickCubeEnv):
    def __init__(self, *args, num_cubes: int = 1, **kwargs):
        self.num_cubes = int(num_cubes)
        assert self.num_cubes >= 1
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # builds table, target red `self.cube`, and goal_site
        super()._load_scene(options)
        self.distractors = []
        for i in range(self.num_cubes - 1):
            color = _DISTRACTOR_COLORS[i % len(_DISTRACTOR_COLORS)]
            d = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=color,
                name=f"distractor_{i}",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
            self.distractors.append(d)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # places table, target cube, goal_site
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            for d in self.distractors:
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                xyz[:, 0] += self.cube_spawn_center[0]
                xyz[:, 1] += self.cube_spawn_center[1]
                xyz[:, 2] = self.cube_half_size
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                d.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            for i, d in enumerate(self.distractors):
                obs[f"distractor{i}_pose"] = d.pose.raw_pose
                obs[f"tcp_to_distractor{i}_pos"] = d.pose.p - self.agent.tcp_pose.p
        return obs
