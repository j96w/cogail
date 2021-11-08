from igibson.envs.igibson_env import iGibsonEnv
import numpy as np
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import ArticulatedObject
import math
import pybullet as p
import torch
import random
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

def create_primitive_shape(mass, shape, dim, color=(0.6, 0, 0, 1), collidable=False, init_xyz=(0, 0, 0.5),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=init_xyz, baseOrientation=init_quat)
    return sid


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
        p.resetBasePositionAndOrientation(self.env.robots[0].robot_ids[0], [-0.8, -0.8, 1.1], quaternion_from_euler(1.1, 0.0, np.pi))
        p.resetBasePositionAndOrientation(self.env.robots[3].robot_ids[0], [0.5, -1.0, 0.1], quaternion_from_euler(-0.5, np.pi, 0.0))
        # self.env.reset()
        self.env.robots[0].base_reset([-0.8, -0.8, 1.1], quaternion_from_euler(1.1, 0.0, np.pi))
        self.env.robots[3].base_reset([0.5, -1.0, 0.1], quaternion_from_euler(-0.3, np.pi, 0.0))
        self.env.simulator_step()
        self.env.robots[0].knee_down()
        self.env.robots[3].knee_down()
        self.env.robots[0].pose_reset([0.2, -0.2, 0.3], [0.0, 0.0, 0.0], 5000.0)
        self.env.robots[3].pose_reset([0.3, 0.0, 1.1], [0.0, 0.0, 0.0], 5000.0)
        self.env.simulator_step()
        self.env.robots[1].base_reset([-0.8, -0.8, 2.2], quaternion_from_euler(0.0, -1.0, np.pi))  # camera 1
        self.env.robots[2].base_reset([0.3, -0.8, 2.2], quaternion_from_euler(np.pi, -1.0, np.pi))  # camera 2
        self.env.simulator_step()
        self.env.robots[0].base_change([-0.8, -0.8, 0.8], quaternion_from_euler(1.1, 0.0, np.pi))
        self.env.robots[3].base_change([0.3, -1.1, 0.03], quaternion_from_euler(-0.5, np.pi, 0.0))
        self.env.simulator_step()
        self.env.robots[0].robot_specific_reset()
        self.env.robots[3].robot_specific_reset()
        self.env.simulator_step()

        obj = YCBObject('006_mustard_bottle')
        self.obj_id = self.env.simulator.import_object(obj)
        p.changeDynamics(self.obj_id, -1, mass=1.0)

        cabinet = '../iGibson/igibson/data/ig_dataset/objects/bottom_cabinet/47235/47235.urdf'
        obj1 = ArticulatedObject(filename=cabinet)
        self.cab_id = obj1.load()
        obj1.set_position([0, 0, 0.9])
        self.handles = [create_primitive_shape(0.00001, p.GEOM_SPHERE, [0.02, 0.0, 0.0], [0, 0.6, 0, 1], False) for _ in range(2)]
        self.env.simulator.load_articulated_object_in_renderer(self.cab_id)
        self.cab_cons = p.createConstraint(
            parentBodyUniqueId=self.cab_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0.6],
            parentFrameOrientation=(0, 0, 0, 1),
            childFrameOrientation=(0, 0, 0, 1),
        )
        p.changeConstraint(self.cab_cons, maxForce=300000.0)
        self.env.simulator_step()

        self.handles_cons = []
        handles_pos = [[-0.43, -0.25, 0.25],
                       [-0.43, -0.25, 0.45]]
        for i in range(4, 6):
            handle_con = p.createConstraint(
                parentBodyUniqueId=self.cab_id,
                parentLinkIndex=i,
                childBodyUniqueId=self.handles[i - 4],
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=handles_pos[i - 4],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=(0, 0, 0, 1),
                childFrameOrientation=(0, 0, 0, 1),
            )
            p.changeConstraint(handle_con, maxForce=1.0)
            self.handles_cons.append(handle_con)
        self.env.simulator_step()

        self.obj_cons = None
        self.handles_cons = []

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

        self.feature_size = 27
        self.seq_length = 10

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        self.observation_space = spaces.Box(-1.0, 1.0, (270, ))
        self.random_seed_space = spaces.Box(-1.0, 1.0, (2, ))
        self.action_space = spaces.Box(-1.0, 1.0, (14, ))

        self.dataset = self.args.gail_experts_dir
        self.dataset_id = [i for i in range(1, self.args.gail_experts_data_length)]
        self.current_step_size = 10
        self.max_step_size = 500
        self.reset_ratio = 1.0

        self.eval_mode = False

    def check_succ(self, obj_pos, drawer_4, drawer_5):
        if not (self.env.robots[0].hold_cons) and not (self.env.robots[3].hold_cons) and not (self.obj_cons) and \
                obj_pos[0] > -0.8 and obj_pos[0] < -0.1 and obj_pos[1] > -0.3 and obj_pos[1] < 0.1 and obj_pos[2] > 0.2 and obj_pos[2] < 1.0 and drawer_4 < 0.01 and drawer_5 < 0.01:
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

        tip_1 = np.array(list(p.getBasePositionAndOrientation(self.env.robots[0].tip)[0]))
        tip_2 = np.array(list(p.getBasePositionAndOrientation(self.env.robots[3].tip)[0]))

        obj_pos = np.array(list(p.getBasePositionAndOrientation(self.obj_id)[0]))
        obj_ori = np.array(list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.obj_id)[1])))

        handle_pos_1 = np.array(list(p.getBasePositionAndOrientation(self.handles[0])[0]))
        handle_pos_2 = np.array(list(p.getBasePositionAndOrientation(self.handles[1])[0]))

        dis_1 = np.linalg.norm(np.array(tip_1) - np.array(obj_pos))
        if dis_1 < 0.06 and (self.env.robots[0].hold_cons is None) and (not self.env.robots[3].holdobject):
            if self.obj_cons:
                p.removeConstraint(self.obj_cons)
                self.obj_cons = None
                self.env.robots[0].hold_target()

        dis_2 = np.linalg.norm(np.array(tip_2) - np.array(obj_pos))
        if dis_2 < 0.06 and (self.env.robots[3].hold_cons is None) and (self.env.robots[0].holdobject):
            self.env.robots[3].hold_target()
            self.env.robots[0].clear()

        if np.linalg.norm(np.array(tip_1) - np.array(handle_pos_2)) < 0.06 and (self.env.robots[0].hold_cons is None) and (self.env.robots[3].holdobject):
            self.env.robots[0].hold_drawer(self.cab_id, 5)
        if np.linalg.norm(np.array(tip_2) - np.array(handle_pos_1)) < 0.06 and (self.env.robots[3].hold_cons is None) and (self.env.robots[0].holdobject):
            self.env.robots[3].hold_drawer(self.cab_id, 4)

        if (self.env.robots[0].holddrawer and not (self.env.robots[3].hold_cons) and not (self.obj_cons) and obj_pos[0] > -0.8 and
            obj_pos[0] < -0.1 and obj_pos[1] > -0.3 and obj_pos[1] < 0.1 and obj_pos[2] > 0.2 and obj_pos[2] < 1.0 and
            p.getJointState(self.cab_id, self.env.robots[0].holddrawer_id)[0] < 0.01) \
                or (self.env.robots[0].holdobject and obj_pos[0] > -0.8 and obj_pos[0] < -0.1 and (handle_pos_1[1] + 0.04) <
                    obj_pos[1] and (handle_pos_1[2] + 0.3) > obj_pos[2] and obj_pos[1] < 0.1 and self.env.robots[3].holddrawer):
            self.env.robots[0].clear()

        if (self.env.robots[3].holddrawer and not (self.env.robots[0].hold_cons) and not (self.obj_cons) and obj_pos[0] > -0.8 and
            obj_pos[0] < -0.1 and obj_pos[1] > -0.3 and obj_pos[1] < 0.1 and obj_pos[2] > 0.2 and obj_pos[2] < 1.0 and
            p.getJointState(self.cab_id, self.env.robots[3].holddrawer_id)[0] < 0.01) \
                or (self.env.robots[3].holdobject and obj_pos[0] > -0.8 and obj_pos[0] < -0.1 and (handle_pos_2[1] + 0.04) <
                    obj_pos[1] and (handle_pos_2[2] + 0.3) > obj_pos[2] and obj_pos[1] < 0.1 and self.env.robots[0].holddrawer):
            self.env.robots[3].clear()

        new_state = np.concatenate((pos_1 - base_1, pos_2 - base_2, obj_pos - pos_1, obj_pos - pos_2, obj_ori, handle_pos_1 - pos_1, handle_pos_1 - pos_2, handle_pos_2 - pos_1, handle_pos_2 - pos_2), axis=0)
        new_state = torch.FloatTensor(np.array([new_state, ]))

        self.padding = torch.cat((self.padding, new_state), dim=0)[1:]
        states = self.padding.view(1, -1)

        succ = self.check_succ(obj_pos, p.getJointState(self.cab_id, 4)[0], p.getJointState(self.cab_id, 5)[0])
        done = self.check_done()
        if succ:
            done = True

        return states, succ, done

    def base_loc(self):
        base_1 = np.array(list(p.getLinkState(self.env.robots[0].robot_ids[0], self.env.robots[0].pelvis_id)[0]))
        base_2 = np.array(list(p.getLinkState(self.env.robots[3].robot_ids[0], self.env.robots[3].pelvis_id)[0]))

        return base_1, base_2


    def reset(self, start_begin=False, provide=False, provide_states=None, provide_joints=None):
        self.env.robots[0].clear()
        self.env.robots[3].clear()
        if self.obj_cons:
            p.removeConstraint(self.obj_cons)
            self.obj_cons = None
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
        for i in range(p.getNumJoints(self.cab_id)):
            p.resetJointState(self.cab_id, i, selected_joints[selected_frame][now], 0.0)
            now += 1

        p.setJointMotorControlArray(self.env.robots[0].robot_ids[0],
                                    [i for i in range(p.getNumJoints(self.env.robots[0].robot_ids[0]))],
                                    p.POSITION_CONTROL, selected_joints[selected_frame][:42], forces=[500.0 for _ in range(p.getNumJoints(self.env.robots[0].robot_ids[0]))])
        p.setJointMotorControlArray(self.env.robots[3].robot_ids[0],
                                    [i for i in range(p.getNumJoints(self.env.robots[3].robot_ids[0]))],
                                    p.POSITION_CONTROL, selected_joints[selected_frame][42:(-p.getNumJoints(self.cab_id))], forces=[500.0 for _ in range(p.getNumJoints(self.env.robots[3].robot_ids[0]))])

        p.resetBasePositionAndOrientation(self.obj_id, selected_states[selected_frame][24:27], p.getQuaternionFromEuler(selected_states[selected_frame][27:30]))
        self.env.simulator_step()

        if abs(selected_states[selected_frame][-3] - 0) < 0.1:
            self.obj_cons = p.createConstraint(
                parentBodyUniqueId=self.obj_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=(0, 0, 0, 1),
                childFrameOrientation=(0, 0, 0, 1),
            )
            obj_pos, obj_ori = self.env.robots[0].get_a_random_target_pos()
            p.changeConstraint(self.obj_cons, obj_pos, obj_ori, maxForce=30000.0)
            self.stage = 0
        elif abs(selected_states[selected_frame][-3] - 1) < 0.1:
            self.env.robots[0].hold_target()
        elif abs(selected_states[selected_frame][-3] - 2) < 0.1:
            self.env.robots[3].hold_target()

        if abs(selected_states[selected_frame][-2] - 0) < 0.1:
            self.env.robots[0].hold_drawer(self.cab_id, 5)
        elif abs(selected_states[selected_frame][-2] - 1) < 0.1:
            self.env.robots[3].hold_drawer(self.cab_id, 4)

        num_j = p.getNumJoints(self.cab_id)
        target_cab = [0.0 for _ in range(num_j)]
        target_force = [50000.0 for _ in range(num_j)]
        target_force[5] = 1.0
        target_force[4] = 1.0
        p.setJointMotorControlArray(self.cab_id, [i for i in range(num_j)], p.POSITION_CONTROL, target_cab, forces=target_force)

        self.env.simulator_step()

        self.step_count = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        return self.get_states()[0], self.random_variable

    def update_step_size(self, already_steps, total_steps):
        self.current_step_size = min(max(int((float(already_steps) / float(600)) * (self.max_step_size)), 10), 500)
        self.reset_ratio = max(1.0 - (float(already_steps) / float(800)), 0.01)

        print("current step size is:", self.current_step_size)
        print("reset ratio is:", self.reset_ratio)

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