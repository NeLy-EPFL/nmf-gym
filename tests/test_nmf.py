import unittest
import numpy as np
import pandas as pd
import pybullet as p
import NeuroMechFly
import tempfile
import shutil
import json
from time import sleep
from pathlib import Path
from farms_container import Container
from nmf_gym.envs.nmf import _NMFSimulation, NMFSimplePositionControlEnv


# Define actuated joints
# This order of joints needs to be consistent across projects
_joints_per_leg = ['Coxa', 'Coxa_roll', 'Coxa_yaw', 'Femur', 'Femur_roll',
            'Tibia', 'Tarsus1']
joints_42dof = [f'joint_{s}{pos}{j}'
                for s in ('L', 'R')
                for pos in ('F', 'M', 'H')
                for j in _joints_per_leg]


class NMFSimulationTestCase(unittest.TestCase):
    def get_new_sim(self, run_time=0.1, time_step=1e-4, sim_options=dict()):
        container = Container(run_time / time_step)
        _sim_options = {
            'headless': False,
            'model_offset': [0., 0., 11.2e-3],
            'run_time': run_time,
            'time_step': time_step,
            'solver_iterations': 100,
            'base_link': 'Thorax',
            'draw_collisions': True,
            'record': False,
            'camera_distance': 4.4,
            'track': False,
            'slow_down': False,
            'sleep_time': 1e-2,
            'rot_cam': False,
            'ground': 'floor',
            'save_frames': False,
        }
        _sim_options.update(sim_options)
        return _NMFSimulation(container, _sim_options, kp=0.4,
                                control_mode='position')
    
    def test_short_run(self):
        sim = self.get_new_sim(run_time=0.01)
        for i, t in enumerate(np.arange(0, sim.run_time, sim.time_step)):
            joint_pos = {name: p.getJointState(sim.animal, jid)[0]
                         for name, jid in sim.joint_id.items()
                         if 'support' not in name}
            sim.step(t, action_dict={'target_positions': joint_pos})
    
    def _move_act_joints(self, sim):
        # act_joints = ['RFCoxa', 'RFFemur', 'RFTibia',
        #               'RMCoxa_roll', 'RMFemur', 'RMTibia',
        #               'RHCoxa_roll', 'RHFemur', 'RHTibia']
        # act_joints = [f'joint_{x}' for x in act_joints]
        _joints = ['Coxa', 'Coxa_roll', 'Coxa_yaw', 'Femur', 'Femur_roll',
                   'Tibia', 'Tarsus1']
        act_joints = [f'joint_R{p}{x}' for p in ['F', 'M', 'H']
                                       for x in _joints]
        init_pos = {
            k: p.getJointState(sim.animal, jid)[0]
            for k, jid in sim.joint_id.items()
        }
        time_grid = np.arange(0, sim.run_time, sim.time_step)
        nframes_per_joint = time_grid.size // len(act_joints)
        max_degs = 30
        rad_step_size = np.deg2rad(max_degs / nframes_per_joint)
        for i, t in enumerate(time_grid):
            _j = i // nframes_per_joint
            if _j == len(act_joints):
                break
            curr_joint = act_joints[_j]
            tgt_pos = (init_pos[curr_joint] +
                       (i % nframes_per_joint) * rad_step_size)
            action_dict = {'target_positions': {curr_joint: tgt_pos}}
            sim.step(i, action_dict=action_dict)

    def test_move_act_joints(self):
        sim = self.get_new_sim(run_time=0.1)
        self._move_act_joints(sim)

    def test_save_frames(self):
        tempdir = tempfile.mkdtemp()
        rec_options = {'save_frames': True, 'results_path': tempdir}
        try:
            sim = self.get_new_sim(run_time=0.1, sim_options=rec_options)
            self._move_act_joints(sim)
            with open(Path(tempdir) / 'fps.json') as f:
                metadata = json.loads(f.read())
            expected_nfiles = int(0.1 * metadata['fps'])
            self.assertEqual(len(list(Path(tempdir).glob('*.png'))),
                             expected_nfiles)
        finally:
            print(f'Temporary file {tempdir} removed')
            shutil.rmtree(tempdir)

    def test_save_video(self):
        rec_options = {'record': True, 'moviename': 'test_movie.mp4'}
        try:
            sim = self.get_new_sim(run_time=0.1, sim_options=rec_options)
            self._move_act_joints(sim)
            sleep(1)
            video_path = Path(rec_options['moviename'])
            self.assertTrue(video_path.is_file())
        finally:
            print(f'Temporary file {video_path} removed')
            video_path.unlink()



class NMFPositionControlEnvTestCase(unittest.TestCase):
    def test_basic(self):
        env = NMFSimplePositionControlEnv(act_joints=joints_42dof,
                                          run_time=0.02, headless=False,
                                          with_ball=False)
        nsteps = 200
        obs_hist = []
        reward_hist = []
        for i in range(nsteps):
            # sleep(0.1)
            tgt_pos = np.deg2rad(10 * i / nsteps)
            obs, reward, is_done, _ = env.step(np.zeros((42,)) * tgt_pos)
            obs_hist.append(obs)
            reward_hist.append(reward)
            self.assertEqual(is_done, i == nsteps - 1)
        # simulation is now finished, stepping again should raise RuntimeError
        self.assertRaises(RuntimeError, env.step, tgt_pos)
    
    def test_basic_10ms(self):
        env = NMFSimplePositionControlEnv(act_joints=joints_42dof,
                                          run_time=2, headless=False,
                                          time_step=0.01, with_ball=False)
        nsteps = 200
        obs_hist = []
        reward_hist = []
        for i in range(nsteps):
            # sleep(0.1)
            tgt_pos = np.deg2rad(10 * i / nsteps)
            obs, reward, is_done, _ = env.step(np.zeros((42,)) * tgt_pos)
            obs_hist.append(obs)
            reward_hist.append(reward)
            self.assertEqual(is_done, i == nsteps - 1)
        # simulation is now finished, stepping again should raise RuntimeError
        self.assertRaises(RuntimeError, env.step, tgt_pos)
    
    def test_human_render(self):
        env = NMFSimplePositionControlEnv(act_joints=joints_42dof,
                                          run_time=0.0005, with_ball=False)
        nsteps = 5
        obs_hist = []
        reward_hist = []
        for i in range(nsteps):
            tgt_pos = np.deg2rad(60 * i / nsteps)
            img = env.render()
            obs, reward, is_done, _ = env.step(np.ones((42,)) * tgt_pos)
            obs_hist.append(obs)
            reward_hist.append(reward)
            self.assertEqual(is_done, i == nsteps - 1)
        # simulation is now finished, stepping again should raise RuntimeError
        self.assertRaises(RuntimeError, env.step, tgt_pos)
        self.assertEqual(img.shape, (768, 1024, 4))
    
    def test_basic_with_ball_unrealistic(self):
        env = NMFSimplePositionControlEnv(act_joints=joints_42dof,
                                          run_time=0.2, headless=False,
                                          with_ball=True)
        nsteps = 2000
        obs_hist = []
        reward_hist = []
        for i in range(nsteps):
            tgt_pos = np.deg2rad(60 * i / nsteps)
            obs, reward, is_done, _ = env.step(np.ones((42,)) * tgt_pos)
            obs_hist.append(obs)
            reward_hist.append(reward)
            self.assertEqual(is_done, i == nsteps - 1)
        # simulation is now finished, stepping again should raise RuntimeError
        self.assertRaises(RuntimeError, env.step, tgt_pos)


if __name__ == '__main__':
    unittest.main()