import copy
import time

from tqdm import trange

import omnigibson as og
from omnigibson.sensors import TiledCamera


class VectorEnvironment:
    def __init__(self, num_envs, config):
        self.num_envs = num_envs
        if og.sim is not None:
            og.sim.stop()

        # First we create the environments. We can't let DummyVecEnv do this for us because of the play call
        # needing to happen before spaces are available for it to read things from.
        self.envs = [
            og.Environment(configs=copy.deepcopy(config), in_vec_env=True)
            for _ in trange(num_envs, desc="Loading environments")
        ]

        self.tiled_camera = TiledCamera(modalities=["rgb", "depth"])

        # Play, and finish loading all the envs
        og.sim.play()
        for env in self.envs:
            env.post_play_load()

    def step(self, actions):
        observations, rewards, terminates, truncates, infos = [], [], [], [], []
        for i, action in enumerate(actions):
            self.envs[i]._pre_step(action)
        og.sim.step()

        tiled_buffer = self.tiled_camera.get_obs()

        rgb_tile = tiled_buffer["rgb"].cpu().numpy()
        depth_tile = tiled_buffer["depth"].cpu().numpy()

        for i, action in enumerate(actions):
            # TODO: ignore camera observation here
            # TODO: potentially, we could get the tiled image first, segment it, and then replace all the normal camera observations with the segmented tiled image
            obs, reward, terminated, truncated, info = self.envs[i]._post_step(action)
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminated)
            truncates.append(truncated)
            infos.append(info)

        return observations, rewards, terminates, truncates, infos

    def reset(self):
        for env in self.envs:
            env.reset()

        # TODO: reset tiled rendering camera

    def close(self):
        pass

    def __len__(self):
        return self.num_envs
