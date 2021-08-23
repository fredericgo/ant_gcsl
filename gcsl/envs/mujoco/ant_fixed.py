import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
xml_file = str(Path(__file__).resolve().parent / "assets" / "ant_fixed.xml")


class Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
    
    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        state = self.state_vector()
        notdone = np.isfinite(state).all() 
        done = not notdone
        ob = self._get_obs()
        return (ob, 0, done, dict(),)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = np.array([0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])
        qpos = np.random.randn(self.model.nq) * .5
        qpos = np.array([np.clip(qpos[i], self.model.jnt_range[i][0], self.model.jnt_range[i][1]) 
                        for i in range(self.model.nq)])
        
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.5
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
