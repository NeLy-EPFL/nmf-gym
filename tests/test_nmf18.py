import unittest
import numpy as np
import pybullet as p
from farms_container import Container
from gym_nmf.envs.nmf18 import _NMF18Simulation

class NMF18SimulationTestCase(unittest.TestCase):
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
            'rot_cam': True,
            'ground': 'floor',
            'save_frames': False,
        }
        _sim_options.update(sim_options)
        return _NMF18Simulation(container, _sim_options, kp=0.4,
                                control_mode='position')
    
    def test_short_run(self):
        sim = self.get_new_sim(run_time=0.01)
        for i, t in enumerate(np.arange(0, sim.run_time, sim.time_step)):
            joint_pos = {name: p.getJointState(sim.animal, jid)[0]
                         for name, jid in sim.joint_id.items()
                         if 'support' not in name}
            sim.step(t, action_dict={'target_positions': joint_pos})
    
    def test_move_act_joints(self):
        sim = self.get_new_sim(run_time=0.1)
        act_joints = ['RFCoxa', 'RFFemur', 'RFTibia',
                      'RMCoxa_roll', 'RMFemur', 'RMTibia',
                      'RHCoxa_roll', 'RHFemur', 'RHTibia']
        act_joints = [f'joint_{x}' for x in act_joints]
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
            sim.step(t, action_dict=action_dict)


if __name__ == '__main__':
    unittest.main()