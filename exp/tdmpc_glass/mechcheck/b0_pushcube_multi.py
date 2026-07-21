"""
B0 task: PushCubeMulti-v1 — LEARNABLE, multi-object, CONFIGURABLE-object-count.

Subclasses ManiSkill's PushCube (push the blue target cube into the goal region
on the table; no lifting required => PPO learns it fast). We add `num_cubes - 1`
distractor cubes whose poses + tcp-relative positions are included in the STATE
obs. The reward/success depend ONLY on the target cube, so the core task is
unchanged across object counts; only obs dimensionality/composition grows.

Object-count knob: env kwarg `num_cubes` (int >= 1).
"""
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

_DISTRACTOR_COLORS = [
    [160, 12, 42, 255], [12, 160, 42, 255], [200, 200, 12, 255],
    [160, 12, 160, 255], [12, 160, 160, 255], [200, 120, 12, 255],
    [120, 120, 120, 255], [80, 30, 150, 255],
]


@register_env("PushCubeMulti-v1", max_episode_steps=50)
class PushCubeMultiEnv(PushCubeEnv):
    def __init__(self, *args, num_cubes: int = 1, **kwargs):
        self.num_cubes = int(num_cubes)
        assert self.num_cubes >= 1
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)  # table, target self.obj, goal_region
        self.distractors = []
        for i in range(self.num_cubes - 1):
            color = np.array(_DISTRACTOR_COLORS[i % len(_DISTRACTOR_COLORS)]) / 255
            d = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=color,
                name=f"distractor_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
            self.distractors.append(d)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            for k, d in enumerate(self.distractors):
                xyz = torch.zeros((b, 3))
                # spread distractors around so they don't all stack on the target
                xyz[..., :2] = torch.rand((b, 2)) * 0.3 - 0.15
                xyz[..., 2] = self.cube_half_size
                q = [1, 0, 0, 0]
                d.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            for i, d in enumerate(self.distractors):
                obs[f"distractor{i}_pose"] = d.pose.raw_pose
                obs[f"tcp_to_distractor{i}_pos"] = d.pose.p - self.agent.tcp.pose.p
        return obs
