import os
import abc
import time
import gym
import numpy as np
import pandas as pd
import pybullet as p
from pathlib import Path
from PIL import Image
from farms_container import Container

import NeuroMechFly
from NeuroMechFly.sdf.units import SimulationUnitScaling
from NeuroMechFly.simulation.bullet_simulation import BulletSimulation


_neuromechfly_path = Path(NeuroMechFly.__path__[0]).parent


class _NMF18Simulation(BulletSimulation):
    def __init__(self, container, sim_options, control_mode, kp=None, kv=None,
                 units=SimulationUnitScaling(meters=1000, kilograms=1000)):

        if 'model' not in sim_options:
            sim_options['model'] = str(_neuromechfly_path /
                'data/design/sdf/neuromechfly_locomotion_optimization.sdf'
            )
        if 'pose' not in sim_options:
            sim_options['pose'] = str(_neuromechfly_path /
                'data/config/pose/pose_tripod.yaml'
            )
        super().__init__(container, units, **sim_options)

        self.kp = kp
        self.kv = kv

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

        self.last_draw = []
        self.control_mode = control_mode
        
    
    def controller_to_actuator(self, t, action_dict):
        if self.control_mode == 'position':
            self._position_controller_to_actuator(t, action_dict)
        else:
            raise NotImplementedError(
                f'Control mode "{self.control_mode}" not implemented')
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
                                    positionGain=self.kp)
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

    def _save_curr_frame(self):
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
        Returns
        -------
        out :
        """
        self._reposition_camera(t)

        if self.save_frames:
            self._save_curr_frame()

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


class NMF18PositionControlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, run_time=2.0, time_step=1e-4, kp=0.4, kv=0.9,
                 headless=True, with_ball=True, sim_options=dict()):
        super().__init__()
        self.run_time = run_time
        self.time_step = time_step
        self.kp = kp
        self.kv = kv
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
            'rot_cam': True,
            'ground': 'floor',
            'save_frames': False,
        }
        self.sim_options.update(sim_options)
        self.sim_options.update({'headless': headless,
                                 'ground': 'ball' if with_ball else 'floor'})
        self.act_joints = ['FCoxa', 'FFemur', 'FTibia',
                           'MCoxa_roll', 'MFemur', 'MTibia',
                           'HCoxa_roll', 'HFemur', 'HTibia']
        self.act_joints = [f'joint_{s}{x}' for s in ['R', 'L']
                                           for x in self.act_joints]
        self.max_niters = int(np.ceil(self.run_time / self.time_step))

        # Define spaces
        # action space: dim=18 (target position for each joint)
        # observ space: dim=40 (pos & vel for each joint, base pos/vel in 2D)
        self.action_space = gym.spaces.box.Box(low=-np.pi, high=np.pi,
                                               shape=(18,))
        self.observation_space = gym.spaces.box.Box(low=-np.pi, high=np.pi,
                                                    shape=(18 * 2 + 4,))

        # Initialize bullet simulation
        self.sim = None
        self.reset()

    def step(self, action):
        if self.curr_iter == self.max_niters:
            raise RuntimeError('Simulation overrun')

        # Parse action
        tgt_pos_dict = self._pos_control_default_pos.copy()
        tgt_pos_dict.update({k: v for k, v in zip(self.act_joints, action)})

        # Step simulation
        self.sim.step(self.curr_time,
                      action_dict={'target_positions': tgt_pos_dict})
        self.curr_time += self.time_step
        self.curr_iter += 1

        # Return observation and reward
        curr_state = self.sim.get_curr_state()
        observ = np.array([
            *[curr_state[joint][0] for joint in self.act_joints],    # position
            *[curr_state[joint][1] for joint in self.act_joints],    # velocity
            *self.sim.base_position, *self.sim.base_linear_velocity    # base
        ])
        reward = self.calculate_reward(observ, tgt_pos_dict)

        is_done = (self.curr_iter == self.max_niters)
        debug_info = dict()

        return observ, reward, is_done, debug_info


    def reset(self):
        del self.sim    # Do this explicitly to avoid physics engine confusion
        container = Container(self.max_niters)
        self.sim = _NMF18Simulation(container, self.sim_options,
                                    kp=self.kp, kv=self.kv,
                                    control_mode='position')
        self.curr_iter = 0
        self.curr_time = 0
        init_state = self.sim.get_curr_state()
        self._pos_control_default_pos = {k: v[0] for k, v in init_state.items()
                                         if k not in self.act_joints}

    def render(self, mode='human'):
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
    
    # @abc.abstractmethod
    def calculate_reward(self, observation, latest_action_dict=None) -> float:
        return NotImplemented