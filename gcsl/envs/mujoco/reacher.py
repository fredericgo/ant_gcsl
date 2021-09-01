import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

xml_file = str(Path(__file__).resolve().parent / "assets" / "reacher.xml")


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, 0, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-1, high=1, size=self.model.nq)
            + self.init_qpos
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                #np.cos(theta),
                #np.sin(theta),
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )