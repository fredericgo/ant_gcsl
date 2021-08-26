import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
xml_file = str(Path(__file__).resolve().parent / "assets" / "ant.xml")


class Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)
    

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.27 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (ob, 0, done, dict(),)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        nq = self.model.nq
        num_jnt = self.model.njnt
        
        q0 = self.init_qpos[:7] + np.random.randn(7) * .5
        qb = np.random.randn(nq - 7) * .5
        qb = [np.clip(qb[i-1], 
                        self.model.jnt_range[i][0], 
                        self.model.jnt_range[i][1]) 
                        for i in range(1, num_jnt)]
        qpos = np.concatenate([q0, qb])
        qvel = self.init_qvel + np.random.randn(self.model.nv) * .5
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * .5
       
