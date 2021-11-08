from igibson.envs.igibson_env import iGibsonEnv
from time import time
import time as t
import igibson
import os
import numpy as np
from igibson.objects.ycb_object import YCBObject
import math
import pybullet as p
import torch
import random
import copy
from gym import spaces


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    _NEXT_AXIS = [1, 2, 0, 1]
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q

def process_rot(rotation):
    if rotation[0] > 0:
        rotation[0] = np.pi - rotation[0]
    else:
        rotation[0] = -np.pi - rotation[0]

    if rotation[2] > 0:
        rotation[2] = np.pi - rotation[2]
    else:
        rotation[2] = -np.pi - rotation[2]

    return np.array([-rotation[2], rotation[1], -rotation[0]])


class igEnv():
    def __init__(self, args):
        self.args = args
        self.config_filename = self.args.ig_config
        self.env = iGibsonEnv(config_file=self.config_filename, mode=self.args.ig_render_mode)
        p.resetBasePositionAndOrientation(self.env.robots[0].robot_ids[0], [-0.75, -0.4, 1.1], quaternion_from_euler(0.0, 0.0, np.pi))
        p.resetBasePositionAndOrientation(self.env.robots[3].robot_ids[0], [0.75, -0.4, 0.1], quaternion_from_euler(0.0, np.pi, 0.0))
        # self.env.reset()
        self.env.robots[0].base_reset([-0.75, -0.4, 1.1], quaternion_from_euler(0.0, 0.0, np.pi))
        self.env.robots[3].base_reset([0.75, -0.4, 0.1], quaternion_from_euler(0.0, np.pi, 0.0))
        self.env.simulator_step()
        self.env.robots[0].pose_reset([0.2, -0.2, 0.3], [0.0, 0.0, 0.0], 5000.0)
        self.env.robots[3].pose_reset([0.4, 0.0, 1.2], [0.0, 0.0, 0.0], 5000.0)
        self.env.simulator_step()
        self.env.robots[1].base_reset([-0.75, -0.4, 2.2], quaternion_from_euler(0.0, -1.0, np.pi))  # camera 1
        self.env.robots[2].base_reset([0.75, -0.4, 2.2], quaternion_from_euler(np.pi, -1.0, np.pi))  # camera 2
        self.env.simulator_step()
        self.env.robots[0].base_change([-0.75, -0.4, 1.1], quaternion_from_euler(0.0, 0.0, np.pi))
        self.env.robots[3].base_change([0.75, -0.4, 0.1], quaternion_from_euler(0.0, np.pi, 0.0))
        self.env.simulator_step()
        self.env.robots[0].robot_specific_reset()
        self.env.robots[3].robot_specific_reset()

        obj = YCBObject('003_cracker_box')
        self.obj_id = self.env.simulator.import_object(obj)
        p.changeDynamics(self.obj_id, -1, mass=0.1)

        self.obj_cons = p.createConstraint(
            parentBodyUniqueId=self.obj_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=(0, 0, 0.707, 0.707),
            childFrameOrientation=(0, 0, 0, 1),
        )
        obj_pos = self.env.robots[0].get_a_random_target_pos()
        p.changeConstraint(self.obj_cons, obj_pos, maxForce=30000.0)

        self.env.robots[0].give_target(self.obj_id)
        self.env.robots[3].give_target(self.obj_id)
        self.env.simulator_step()

        self.step_count = 0

        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.FloatTensor(np.array([[i, j] for i in self.linspace for j in self.linspace])).view(-1, 2)
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.feature_size = 15
        self.seq_length = 10

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        self.observation_space = spaces.Box(-1.0, 1.0, (150, ))
        self.random_seed_space = spaces.Box(-1.0, 1.0, (2, ))
        self.action_space = spaces.Box(-1.0, 1.0, (14, ))

        self.dataset = self.args.gail_experts_dir
        self.dataset_id = [i for i in range(1, 117)] + [i for i in range(118, 149)] + [i for i in range(150, 156)]

        self.current_step_size = 10
        self.max_step_size = 300
        self.reset_ratio = 0.01

        self.eval_mode = False

    def check_succ(self):
        if self.env.robots[3].hold_cons:
            pos_1 = np.array(p.getLinkState(self.env.robots[0].robot_ids[0], self.env.robots[0].hand_id)[0])
            pos_2 = np.array(p.getLinkState(self.env.robots[3].robot_ids[0], self.env.robots[3].hand_id)[0])
            if np.linalg.norm(pos_1 - pos_2) > 0.3:
                return True
        return False

    def check_done(self):
        pos = p.getBasePositionAndOrientation(self.obj_id)[0]
        return pos[2] < 0.2

    def get_states(self):
        base_1 = np.array(list(p.getLinkState(self.env.robots[0].robot_ids[0], self.env.robots[0].pelvis_id)[0]))
        base_2 = np.array(list(p.getLinkState(self.env.robots[3].robot_ids[0], self.env.robots[3].pelvis_id)[0]))

        pos_1 = np.array(list(p.getLinkState(self.env.robots[0].robot_ids[0], self.env.robots[0].hand_id)[0]))
        pos_2 = np.array(list(p.getLinkState(self.env.robots[3].robot_ids[0], self.env.robots[3].hand_id)[0]))

        tip_1 = list(p.getBasePositionAndOrientation(self.env.robots[0].tip)[0])
        tip_2 = list(p.getBasePositionAndOrientation(self.env.robots[3].tip)[0])

        # ori_1 = np.array(list(self.env.robots[0].current_hand_ori))
        # ori_2 = np.array(list(self.env.robots[3].current_hand_ori))

        obj_pos = np.array(list(p.getBasePositionAndOrientation(self.obj_id)[0]))
        obj_ori = np.array(list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_id)[1])))

        dis_1 = np.linalg.norm(np.array(tip_1) - np.array(obj_pos))
        if dis_1 < 0.06 and self.env.robots[0].hold_cons == None:
            if self.obj_cons:
                p.removeConstraint(self.obj_cons)
                self.obj_cons = None
                self.env.robots[0].hold_target()

        dis_2 = np.linalg.norm(np.array(tip_2) - np.array(obj_pos))
        if dis_2 < 0.06 and self.env.robots[3].hold_cons == None and self.env.robots[3].gripper == self.env.robots[3].gripper_close:
            if self.env.robots[0].hold_cons:
                self.env.robots[3].hold_target()
                self.env.robots[0].clear()

        new_state = np.concatenate((pos_1 - base_1, pos_2 - base_2, obj_pos - pos_1, obj_pos - pos_2, obj_ori), axis=0)
        new_state = torch.FloatTensor(np.array([new_state, ]))

        self.padding = torch.cat((self.padding, new_state), dim=0)[1:]
        states = self.padding.view(1, -1)

        succ = self.check_succ()
        done = self.check_done()
        if succ:
            done = True

        return states, succ, done

    def base_loc(self):
        base_1 = np.array(list(p.getLinkState(self.env.robots[0].robot_ids[0], self.env.robots[0].pelvis_id)[0]))
        base_2 = np.array(list(p.getBasePositionAndOrientation(self.env.robots[3].robot_ids[0])[0]))

        return base_1, base_2


    def reset(self, start_begin=False, provide=False, provide_states=None, provide_joints=None):
        self.env.robots[0].clear()
        self.env.robots[3].clear()
        if self.obj_cons:
            p.removeConstraint(self.obj_cons)
            self.obj_cons = None
        self.env.simulator_step()

        self.env.robots[0].pose_reset([0.2, -0.2, 0.3], [0.0, 0.0, 0.0], 5000.0)
        self.env.robots[3].pose_reset([0.4, 0.0, 1.2], [0.0, 0.0, 0.0], 5000.0)
        p.resetBasePositionAndOrientation(self.obj_id, [0.0, 0.0, 0.2], [0, 0, 0.707, 0.707])
        self.env.simulator_step()

        self.env.robots[0].give_target(self.obj_id)
        self.env.robots[3].give_target(self.obj_id)

        if provide:
            selected_states = provide_states
            selected_joints = provide_joints
        else:
            selected_id = random.choice(self.dataset_id)
            selected_states = np.load('{0}/{1}_states.npy'.format(self.dataset, selected_id))
            selected_joints = np.load('{0}/{1}_joints.npy'.format(self.dataset, selected_id))

        if start_begin or self.eval_mode:
            selected_frame = 0
        else:
            selected_frame = random.randint(0, max(1, int(self.reset_ratio * float(len(selected_states)-1))))

        self.env.robots[0].give_initial_ee_pose(selected_states[selected_frame][12:15], selected_states[selected_frame][15:18])
        self.env.robots[3].give_initial_ee_pose(selected_states[selected_frame][18:21], selected_states[selected_frame][21:24])

        now = 0
        for i in range(p.getNumJoints(self.env.robots[0].robot_ids[0])):
            p.resetJointState(self.env.robots[0].robot_ids[0], i, selected_joints[selected_frame][now], 0.0)
            now += 1
        for i in range(p.getNumJoints(self.env.robots[3].robot_ids[0])):
            p.resetJointState(self.env.robots[3].robot_ids[0], i, selected_joints[selected_frame][now], 0.0)
            now += 1
        p.setJointMotorControlArray(self.env.robots[0].robot_ids[0],
                                    [i for i in range(p.getNumJoints(self.env.robots[0].robot_ids[0]))],
                                    p.POSITION_CONTROL, selected_joints[selected_frame][:42], forces=[500.0 for _ in range(p.getNumJoints(self.env.robots[0].robot_ids[0]))])

        p.setJointMotorControlArray(self.env.robots[3].robot_ids[0],
                                    [i for i in range(p.getNumJoints(self.env.robots[3].robot_ids[0]))],
                                    p.POSITION_CONTROL, selected_joints[selected_frame][42:], forces=[500.0 for _ in range(p.getNumJoints(self.env.robots[3].robot_ids[0]))])
        p.resetBasePositionAndOrientation(self.obj_id, selected_states[selected_frame][-7:-4], p.getQuaternionFromEuler(selected_states[selected_frame][-4:-1]))
        self.env.simulator_step()

        if abs(selected_states[selected_frame][-1] - 0) < 0.1:
            self.obj_cons = p.createConstraint(
                parentBodyUniqueId=self.obj_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=(0, 0, 0.707, 0.707),
                childFrameOrientation=(0, 0, 0, 1),
            )
            obj_pos = self.env.robots[0].get_a_random_target_pos()
            p.changeConstraint(self.obj_cons, obj_pos, maxForce=30000.0)
            self.stage = 0
        else:
            if abs(selected_states[selected_frame][-1] - 1) < 0.1:
                self.env.robots[0].hold_target()
                self.stage = 1
            else:
                self.env.robots[3].hold_target()
                self.stage = 2

        self.env.simulator_step()

        self.step_count = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        return self.get_states()[0], self.random_variable

    def start_eval(self):
        self.eval_mode = True
        self.current_step_size = self.max_step_size

    def stop_eval(self):
        self.eval_mode = False

    def step(self, actions):
        success = False
        if self.step_count > self.current_step_size:
            done = True
            states, _ = self.reset()
            return states, torch.FloatTensor([[float(success)], ]), [done], [{'bad_transition':True}], self.random_variable
        else:
            action = np.clip(actions[0].detach().cpu().numpy(), -1.0, 1.0)

            action_human = action[:7]
            action_human[:3] /= 1000.0
            rotation_human = np.clip(action_human[3:6], -0.9, 0.9)
            gripper_human = action_human[-1:]

            action_robot = action[7:14]
            action_robot[:3] /= 1000.0
            rotation_robot = np.clip(action_robot[3:6], -0.9, 0.9)
            gripper_robot = action_robot[-1:]

            del_action = np.array([0.0, 0.0, 0.0, float(action_human[0]), float(action_human[1]), float(action_human[2]), 0.0, 0.0, 0.0, 0.0]) * 6.0
            del_action[6:-1] = rotation_human
            del_action[-1] = gripper_human

            del_action_2 = np.array([0.0, 0.0, 0.0, float(action_robot[0]), float(action_robot[1]), float(action_robot[2]), 0.0, 0.0, 0.0, 0.0]) * 6.0
            del_action_2[6:-1] = rotation_robot
            del_action_2[-1] = gripper_robot

            self.env.step(del_action, del_action_2)
            self.step_count += 1
            states, succ, done = self.get_states()

            if succ:
                success = True
                done = True
                states, _ = self.reset(True)
                return states, torch.FloatTensor([[float(success)],]), [done], [{}], self.random_variable
            elif done:
                states, _ = self.reset()
                return states, torch.FloatTensor([[float(success)], ]), [done], [{'bad_transition':True}], self.random_variable
            else:
                return states, torch.FloatTensor([[float(success)], ]), [done], [{}], self.random_variable