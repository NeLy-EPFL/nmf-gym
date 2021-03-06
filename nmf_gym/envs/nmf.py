from ntpath import join
import os
import abc
import time
import nmf_gym
import gym
import json
import numpy as np
import pandas as pd
import pybullet as p
import farms_pylog as pylog
from uuid import uuid4
from typing import Tuple, Dict
from pathlib import Path
from PIL import Image
from farms_container import Container as _Container

import NeuroMechFly
from NeuroMechFly.sdf.sdf import ModelSDF
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation


_nmf_gym_path = Path(nmf_gym.__path__[0]).parent
_fixed_positions = {
        'joint_A3': -15,
        'joint_A4': -15,
        'joint_A5': -15,
        'joint_A6': -15,
        'joint_LAntenna': 35,
        'joint_RAntenna': -35,
        'joint_Rostrum': 90,
        'joint_Haustellum': -60,
        'joint_LWing_roll': 90,
        'joint_LWing_yaw': -17,
        'joint_RWing_roll': -90,
        'joint_RWing_yaw': 17,
        'joint_Head': 10
    }
_fixed_positions = {k: np.deg2rad(v) for k, v in _fixed_positions.items()}

def load_joint_limit(sdf_path):
    _sdf_model = ModelSDF.read(sdf_path)[0]
    joint_limits = {
        joint.name: joint.axis.limits[:2]
        for joint in _sdf_model.joints
        if joint.axis.limits
    }
    return joint_limits


class Container(_Container):
    """Extend FARMS Container to make maximum iterations accessible."""
    @property
    def max_iterations(self):
        return int(self.__max_iterations)


class _NMFSimulation(BulletSimulation):
    def __init__(self, container, sim_options, control_mode, kp=None, kv=None,
                 max_force=None, fixed_positions=dict(),
                 units=SimulationUnitScaling(meters=1000, kilograms=1000)):

        if 'model' not in sim_options:
            # Joint limits strictly enforced (joint types = revolute in SDF)
            sim_options['model'] = str(_nmf_gym_path /
                'data/design/sdf/neuromechfly_42dof_with_limit.sdf'
            )
        if 'pose' not in sim_options:
            sim_options['pose'] = str(_nmf_gym_path /
                'data/pose/pose_stretch.yaml'
            )
        if sim_options['record']:
            # PyBullet only accept filenames, not POSIX paths. Annoying hack.
            # Also, PyBullet outputs video with wrong fps, hardcoded hacky fix
            sim_options['moviespeed'] = sim_options.get('moviespeed', 1) / 11.06
        if 'results_path' in sim_options:
            sim_options['results_path'] = str(sim_options['results_path'])
        self.fixed_positions = fixed_positions
        super().__init__(container, units, **sim_options)

        self.kp = kp
        self.kv = kv
        self.max_force = np.inf if max_force is None else max_force

        # Set the physical properties of the environment
        dynamics = {
            "lateralFriction": 1.0,
            "restitution": 0.0,
            "spinningFriction": 0.0,
            "rollingFriction": 0.0,
            "linearDamping": 0.0,
            "angularDamping": 0.0,
            "maxJointVelocity": 1e8}
        for _link, idx in self.link_id.items():
            for name, value in dynamics.items():
                p.changeDynamics(self.animal, idx, **{name: value})

        # Handle save frames
        if self.save_frames:
            # Save only this number of frames per second
            tgt_video_fps = sim_options.get('target_video_fps', 30)
            # This is actually offset a bit to make it divisible
            real_rec_interval = (1 / tgt_video_fps) / self.time_step
            self._real_rec_interval = int(np.rint(real_rec_interval))
            # Save this number to the output dir as metadata
            with open(Path(self.path_imgs) / 'fps.json', 'w') as f:
                metadata = {
                    'fps': 1 / (self._real_rec_interval * self.time_step),
                    'interval': self._real_rec_interval
                }
                f.write(json.dumps(metadata))

        self.last_draw = []
        self.control_mode = control_mode
        
    
    def controller_to_actuator(self, t, action_dict):
        if self.control_mode == 'position':
            self._position_controller_to_actuator(t, action_dict)
        else:
            raise NotImplementedError(
                f'Control mode "{self.control_mode}" not implemented')
        
        # Now, handle fixed positions via direct pos control
        for joint_name, position in self.fixed_positions.items():
            jid = self.joint_id[joint_name]
            p.setJointMotorControl2(self.animal, jid,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=position,
                                    positionGain=self.kp,
                                    force=self.max_force)
        
        self._mark_collisions()
    
    def _position_controller_to_actuator(self, t, action_dict):
        tgt_positions = {k: p.getJointState(self.animal, jid)[0]
                         for k, jid in self.joint_id.items()}
        tgt_positions.update(action_dict['target_positions'])
        for joint_name, position in tgt_positions.items():
            jid = self.joint_id[joint_name]
            p.setJointMotorControl2(self.animal, jid,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=position,
                                    positionGain=self.kp,
                                    force=self.max_force)
            p.changeDynamics(self.animal, jid, maxJointVelocity=1e8)

    def _change_color(self, identity, color):
        """ Change color of a given body segment. """
        p.changeVisualShape(
            self.animal,
            self.link_id[identity],
            rgbaColor=color)
    
    def _mark_collisions(self):
        """ Change the color of the colliding body segments. """
        if self.draw_collisions:
            draw = []
            if self.behavior == 'walking':
                links_contact = self.get_current_contacts()
                link_names = list(self.link_id.keys())
                link_ids = list(self.link_id.values())
                for i in links_contact:
                    link1 = link_names[link_ids.index(i)]
                    if link1 not in draw:
                        draw.append(link1)
                        self._change_color(link1, self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        self._change_color(link, self.color_legs)

            elif self.behavior == 'grooming':
                # Don't consider the ground sensors
                collision_forces = self.contact_normal_force[len(
                    self.ground_contacts):, :]
                links_contact = np.where(
                    np.linalg.norm(collision_forces, axis=1) > 0
                )[0]
                for i in links_contact:
                    link1 = self.self_collisions[i][0]
                    link2 = self.self_collisions[i][1]
                    if link1 not in draw:
                        draw.append(link1)
                        self._change_color(link1, self.color_collision)
                    if link2 not in draw:
                        draw.append(link2)
                        self._change_color(link2, self.color_collision)
                for link in self.last_draw:
                    if link not in draw:
                        if 'Antenna' in link:
                            self._change_color(link, self.color_body)
                        else:
                            self._change_color(link, self.color_legs)
            self.last_draw = draw
    
    def _reposition_camera(self, t):
        # Camera
        if self.gui == p.GUI and self.track_animal:
            base = np.array(self.base_position) * self.units.meters
            base[2] = self.model_offset[2]
            yaw = 30
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance, yaw, pitch, base
            )

        # Walking camera sequence, set rotate_camera to True to activate
        if ((self.gui == p.GUI) and
                (self.rotate_camera) and
                (self.behavior == 'walking')):
            base = np.array(self.base_position) * self.units.meters

            if t < 3 / self.time_step:
                yaw = 0
                pitch = -10
            elif t >= 3 / self.time_step and t < 4 / self.time_step:
                yaw = (t - (3 / self.time_step)) / (1 / self.time_step) * 90
                pitch = -10
            elif t >= 4 / self.time_step and t < 4.25 / self.time_step:
                yaw = 90
                pitch = -10
            elif t >= 4.25 / self.time_step and t < 4.75 / self.time_step:
                yaw = 90
                pitch = (t - (4.25 / self.time_step)) / \
                    (0.5 / self.time_step) * 70 - 10
            elif t >= 4.75 / self.time_step and t < 5 / self.time_step:
                yaw = 90
                pitch = 60
            elif t >= 5 / self.time_step and t < 5.5 / self.time_step:
                yaw = 90
                pitch = 60 - (t - (5 / self.time_step)) / \
                    (0.5 / self.time_step) * 70
            elif t >= 5.5 / self.time_step and t < 7 / self.time_step:
                yaw = (t - (5.5 / self.time_step)) / \
                    (1.5 / self.time_step) * 300 + 90
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        # Grooming camera sequence, set rotate_camera to True to activate
        if ((self.gui == p.GUI) and
                (self.rotate_camera) and
                (self.behavior == 'grooming')):
            base = np.array(self.base_position) * self.units.meters
            if t < 0.25 / self.time_step:
                yaw = 0
                pitch = -10
            elif t >= 0.25 / self.time_step and t < 2.0 / self.time_step:
                yaw = (t - (0.25 / self.time_step)) / \
                    (1.75 / self.time_step) * 150
                pitch = -10
            elif t >= 2.0 / self.time_step and t < 3.5 / self.time_step:
                yaw = 150 - (t - (2.0 / self.time_step)) / \
                    (1.5 / self.time_step) * 120
                pitch = -10
            else:
                yaw = 30
                pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

        if self.gui == p.GUI and self.rotate_camera and self.behavior is None:
            base = np.array(self.base_position) * self.units.meters
            yaw = (t - (self.run_time / self.time_step)) / \
                (self.run_time / self.time_step) * 360
            pitch = -10
            p.resetDebugVisualizerCamera(
                self.camera_distance,
                yaw,
                pitch,
                base)

    def _save_curr_frame(self, t):
        if self.gui == p.DIRECT:
            base = np.array(self.base_position) * self.units.meters
            matrix = p.computeViewMatrixFromYawPitchRoll(
                base, self.camera_distance, 5, -10, 0, 2
            )
            projectionMatrix = [1.0825318098068237, 0.0, 0.0, 0.0, 0.0,
                                1.732050895690918, 0.0, 0.0, 0.0, 0.0,
                                -1.0002000331878662, -1.0, 0.0, 0.0,
                                -0.020002000033855438, 0.0]
            img = p.getCameraImage(1024, 768, viewMatrix=matrix,
                                   projectionMatrix=projectionMatrix)
        if self.gui == p.GUI:
            img = p.getCameraImage(
                1024, 768, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = img[2]
        im = Image.fromarray(rgb_array)

        im_name = f"{self.path_imgs}/Frame_{t:06d}.png"
        if not os.path.exists(self.path_imgs):
            os.mkdir(self.path_imgs)

        im.save(im_name)

        # disable rendering temporary makes adding objects faster
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    def step(self, t, action_dict=dict(), optimization=False):
        """ Step the simulation.
        t (int)
            This is the current iteration! Not actual time
        Returns
        -------
        out :
        """
        self._reposition_camera(t)

        if self.save_frames and t % self._real_rec_interval == 0:
            self._save_curr_frame(t)

        # Update logs
        self.update_logs()
        # Update container log
        self.container.update_log()
        # Update the feedback to controller
        self.feedback_to_controller()
        # Step controller
        if self.controller_config:
            self.controller.step(self.time_step)
        # Update the controller_to_actuator
        self.controller_to_actuator(t, action_dict)
        # Step muscles
        if self.use_muscles:
            self.muscles.step()
        # Step time
        self.time += self.time_step
        # Step physics
        solver = p.stepSimulation()

        # Slow down the simulation
        if self.slow_down:
            time.sleep(self.sleep_time)
        # Check if optimization is to be killed
        if optimization:
            optimization_status = self.optimization_check()
            return optimization_status
        return True

    def feedback_to_controller(self):
        pass

    def optimization_check(self):
        pass

    def update_parameters(self, params):
        pass

    def run(self, optimization=False):
        raise RuntimeError(
            'Do not interact with NMF18Simulation.run directly. Interaction '
            'should be rolled out step by step.'
        )
    
    def get_curr_state(self, joints=None):
        if not joints:
            joints = self.joint_id.keys()
        curr_state = {}
        for joint_name in joints:
            jid = self.joint_id[joint_name]
            curr_state[joint_name] = p.getJointState(self.animal, jid)
        return curr_state
    
    @property
    def virtual_position(self):
        if self.ground == 'ball':
            rot = np.array(self.ball_rotations)
        else:
            rot = np.zeros((3,))
        return rot * self.ball_radius * self.units.meters
    
    @property
    def virtual_velocity(self):
        if self.ground == 'ball':
            drot_dt = np.array(self.ball_velocity)
        else:
            drot_dt = np.zeros((3,))
        return drot_dt * self.ball_radius * self.units.meters


class NMFPositionControlBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, act_joints,
                 run_time=2.0, time_step=1e-4, kp=0.4, kv=0.9, max_force=20,
                 headless=True, with_ball=True, sim_options=dict(),
                 set_natural_pose=False, fixed_positions=dict()):
        super().__init__()
        self.run_time = run_time
        self.time_step = time_step
        self.kp = kp
        self.kv = kv
        self.max_force = max_force
        self.sim_options = {
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
        if set_natural_pose:
            fp = _fixed_positions.copy()
            fp.update(fixed_positions)  # user provided vals override defaults
            fixed_positions = fp
            sim_options['model'] = str(_nmf_gym_path /
                'data/design/sdf/neuromechfly_42dof_with_limit_viz.sdf'
            )
        self.fixed_positions = fixed_positions
        self.sim_options.update(sim_options)
        self.sim_options.update({'headless': headless,
                                 'ground': 'ball' if with_ball else 'floor'})
        self.act_joints = act_joints
        self.max_niters = int(np.ceil(self.run_time / self.time_step))

        # Define spaces
        self.action_space, self.observation_space = self._define_spaces()
        
        # Keep a record of pos and vel outside FARMS to make the whole
        # history accessible. Initialized in self.reset
        self.pos_hist = None
        self.vel_hist = None

        # Initialize bullet simulation
        self.sim = None
        self.reset()

        # Get joint limits (lower lim = self.joint_limits[0], upper limits [1])
        # do this after self.reset() so sim.model is defined
        limits_d = load_joint_limit(self.sim.model)
        self.joint_limits = np.array([limits_d[x] for x in self.act_joints]).T
        for i, joint_name in enumerate(self.act_joints):
            link_name = joint_name.replace('joint_', '')
            p.changeDynamics(self.sim.animal, self.sim.link_id[link_name],
                             jointLowerLimit=self.joint_limits[0, i],
                             jointUpperLimit=self.joint_limits[1, i],
                             jointLimitForce=1e8)


    def step(self, action):
        if self.curr_iter == self.max_niters:
            raise RuntimeError('Simulation overrun')

        # Step simulation
        tgt_pos_dict = self._parse_action(action)

        self.sim.step(self.curr_iter,
                      action_dict={'target_positions': tgt_pos_dict})
        self.curr_time += self.time_step
        self.curr_iter += 1

        # Get state from simulation
        curr_state = self.sim.get_curr_state(self.act_joints)
        self.pos_hist[self.curr_iter, :] = [curr_state[k][0]
                                            for k in self.act_joints]
        self.vel_hist[self.curr_iter, :] = [curr_state[k][1]
                                            for k in self.act_joints]

        # Calculate observation and reward
        # Uncomment the following code for out-of-range detection
        observ = self._parse_observation(curr_state)
        reward = self._calculate_reward(observ, tgt_pos_dict)
        # joint_out_of_bound = np.any([
        #     self.pos_hist[self.curr_iter] < self.joint_limits[0] - 0.3,
        #     self.pos_hist[self.curr_iter] > self.joint_limits[1] + 0.3
        # ])
        # if joint_out_of_bound and self.curr_iter > 10:
        #     print(f'OUT OF BOUND at iter {self.curr_iter}=====')
        #     curr_pos = self.pos_hist[self.curr_iter]
        #     for i, joint in enumerate(self.act_joints):
        #         if ((curr_pos[i] < self.joint_limits[0, i] - 0.3) or
        #                 (curr_pos[i] > self.joint_limits[1, i] + 0.3)):
        #             print('%s=%.2f, tgt=%.2f, range=(%.2f, %.2f)'
        #                       % (joint, curr_pos[i], new_pos_vec[i],
        #                          *self.joint_limits[:, i]))
        #     reward -= 10
        is_done = (self.curr_iter == self.max_niters)
        # is_done |= (joint_out_of_bound and self.curr_iter > 10)
        debug_info = dict()

        # print('>>> OBSERV', observ)
        # print('>>> REARD', reward)
        # print('>>> IS_DONE', is_done)
        # assert False
        # from time import sleep
        # sleep(0.05)

        return observ, reward, is_done, debug_info


    def reset(self):
        del self.sim    # Do this explicitly to avoid physics engine confusion
        # Keep a record of pos and vel outside FARMS to make the whole
        # history accessible
        self.pos_hist = np.full((self.max_niters + 1, len(self.act_joints)),
                                np.nan)
        self.vel_hist = self.pos_hist.copy()

        container = Container(self.max_niters)
        self.sim = _NMFSimulation(container, self.sim_options,
                                    kp=self.kp, kv=self.kv,
                                    max_force=self.max_force,
                                    control_mode='position',
                                    fixed_positions=self.fixed_positions)
        self.curr_iter = 0
        self.curr_time = 0
        init_state = self.sim.get_curr_state()
        self.pos_hist[0, :] = [init_state[k][0] for k in self.act_joints]
        self.vel_hist[0, :] = [init_state[k][1] for k in self.act_joints]
        self._pos_control_default_pos = {k: v[0] for k, v in init_state.items()
                                         if k not in self.act_joints}
        self._last_position = 0
        return self._parse_observation(init_state)

    def render(self, mode='human'):
        """
        Note that rendering can be done by setting the `headless`
        parameter to False upon init. This is faster as PyBullet's
        built in rendering code will be invoked.
        """
        if self.sim_options['headless']:
            base = np.array(self.sim.base_position) * self.sim.units.meters
            matrix = p.computeViewMatrixFromYawPitchRoll(
                base, self.sim.camera_distance, 5, -10, 0, 2
            )
            projectionMatrix = [1.0825318098068237, 0.0, 0.0, 0.0, 0.0,
                                1.732050895690918, 0.0, 0.0, 0.0, 0.0,
                                -1.0002000331878662, -1.0, 0.0, 0.0,
                                -0.020002000033855438, 0.0]
            img = p.getCameraImage(1024, 768, viewMatrix=matrix,
                                   projectionMatrix=projectionMatrix)[2]
        else:
            img = p.getCameraImage(1024, 768,
                                   renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

        return img

    def close(self):
        del self.sim
    
    @abc.abstractmethod
    def _define_spaces(self) -> Tuple[gym.Space, gym.Space]:
        """
        Return the Gym action space and the observation space.
        """
        return NotImplemented, NotImplemented

    @abc.abstractmethod
    def _parse_observation(self, curr_state) -> np.ndarray:
        """
        Return the observation in accordance with the observation
        space specs, given the current state returned by
        `_NMF18Simulation.get_curr_state()`.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def _parse_action(self, action) -> Dict[str, float]:
        """
        Parse the given action, in accordance with the action space
        specs, into a dictionary mapping actuated joint names to
        the target positions.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def _calculate_reward(self, observation, latest_action_dict=None) -> float:
        """
        Calculate reward value given the current observation and the
        latest action dict (as returned by `self._parse_action()`).
        """
        return NotImplemented


class NMFSimplePositionControlEnv(NMFPositionControlBaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def _define_spaces(self):
        action_space = gym.spaces.box.Box(low=-np.pi, high=np.pi, shape=(18,))
        obs_space = gym.spaces.box.Box(low=np.deg2rad(-36),
                                       high=np.deg2rad(36),
                                       shape=(18 * 3 + 6,))
        return action_space, obs_space

    def _parse_action(self, action):
        tgt_pos_dict = self._pos_control_default_pos.copy()
        tgt_pos_dict.update({k: v for k, v in zip(self.act_joints, action)})
        return tgt_pos_dict

    def _parse_observation(self, curr_state):
        observ = np.array([
            *[curr_state[joint][0] for joint in self.act_joints],    # position
            *[curr_state[joint][1] for joint in self.act_joints],    # velocity
            *[curr_state[joint][3] for joint in self.act_joints],    # torque
            *self.sim.virtual_position, *self.sim.virtual_velocity   # fly x, x'
        ])
        return observ
    
    def _calculate_reward(self, observation, latest_action_dict):
        return np.nan


class NMFPos2PosDistanceEnv(NMFPositionControlBaseEnv):
    def __init__(self, state_indices, run_time=2, time_step=0.0001,
                 kp=0.4, kv=0.9, max_force=20, headless=True, with_ball=True,
                 sim_options=dict()):
        self.state_indices = state_indices
        super().__init__(run_time, time_step, kp, kv, max_force, headless,
                         with_ball, sim_options)
    
    def _define_spaces(self):
        act_space = gym.spaces.box.Box(low=-np.pi, high=np.pi, shape=(18,))
        obs_space = gym.spaces.box.Box(low=np.deg2rad(-0.02),
                                       high=np.deg2rad(0.02),
                                       shape=(len(self.state_indices), 18))
        return act_space, obs_space
    
    def _parse_action(self, action):
        tgt_pos_dict = self._pos_control_default_pos.copy()
        new_pos_vec = self.pos_hist[self.curr_iter, :] + action
        new_pos_vec = np.maximum(new_pos_vec, self.joint_limits[0])  # lower lim
        new_pos_vec = np.minimum(new_pos_vec, self.joint_limits[1])  # upper lim
        tgt_pos_dict.update({k: v
                             for k, v in zip(self.act_joints, new_pos_vec)})
        return tgt_pos_dict
    
    def _parse_observation(self, curr_state):
        obs = []
        for offset in self.state_indices:
            idx = self.curr_iter + offset
            if idx < 0:
                obs.append(self.pos_hist[0, :])
            else:
                obs.append(self.pos_hist[idx, :])
        return np.array(obs)

    def _calculate_reward(self, observation, latest_action_dict):
        displacement = self.sim.virtual_position[0] - self._last_position
        self._last_position = self.sim.virtual_position[0]
        return displacement