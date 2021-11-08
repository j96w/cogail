import numpy as np
import torch
import torch.utils.data

class ig_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data_ids = [i for i in range(1, 117)] + [i for i in range(118, 149)] + [i for i in range(150, 156)]
        print(self.data_ids)

        self.args = args

        self.data_buff_states = {}
        self.data_buff_actions = {}

        self.video_idx_list = []
        self.frame_idx_list = []

        self.action_mean = []
        self.action_max = []
        self.action_min = []
        self.steps_size = []

        self.seq_length = 10
        self.state_size = 15

        self.dataset_name = self.args.gail_experts_dir

        for item in self.data_ids:
            self.data_buff_states[item] = np.load('{0}/{1}_states.npy'.format(self.dataset_name, item))
            self.data_buff_actions[item] = np.load('{0}/{1}_actions.npy'.format(self.dataset_name, item))

            self.action_mean.append(np.mean(self.data_buff_actions[item], axis=0))
            self.action_max.append(np.max(self.data_buff_actions[item], axis=0))
            self.action_min.append(np.min(self.data_buff_actions[item], axis=0))
            self.steps_size.append(len(self.data_buff_states[item]))

            self.video_idx_list += [item for i in range(1, len(self.data_buff_states[item]) - self.seq_length - 1)]
            self.frame_idx_list += [i for i in range(1, len(self.data_buff_states[item]) - self.seq_length - 1)]

        self.length = len(self.video_idx_list)


    def __getitem__(self, index):

        video_idx = self.video_idx_list[index]
        frame_idx = self.frame_idx_list[index]

        if frame_idx >= (self.seq_length - 1):
            inputs_raw = np.array(self.data_buff_states[video_idx][(frame_idx + 1 - self.seq_length):(frame_idx + 1)])
            base_1 = inputs_raw[:, :3]
            base_2 = inputs_raw[:, 3:6]
            pos_1 = inputs_raw[:, 6:9]
            pos_2 = inputs_raw[:, 9:12]
            cut_pos_1 = inputs_raw[:, 12:15]
            cut_ori_1 = inputs_raw[:, 15:18]
            cut_pos_2 = inputs_raw[:, 18:21]
            cut_ori_2 = inputs_raw[:, 21:24]
            obj_pos = inputs_raw[:, 24:27]
            obj_ori = inputs_raw[:, 27:30]

            inputs = np.concatenate((pos_1 - base_1, pos_2 - base_2, obj_pos - pos_1, obj_pos - pos_2, obj_ori), axis=1)

        else:
            inputs_raw = np.array(self.data_buff_states[video_idx][:(frame_idx + 1)])
            base_1 = inputs_raw[:, :3]
            base_2 = inputs_raw[:, 3:6]
            pos_1 = inputs_raw[:, 6:9]
            pos_2 = inputs_raw[:, 9:12]
            cut_pos_1 = inputs_raw[:, 12:15]
            cut_ori_1 = inputs_raw[:, 15:18]
            cut_pos_2 = inputs_raw[:, 18:21]
            cut_ori_2 = inputs_raw[:, 21:24]
            obj_pos = inputs_raw[:, 24:27]
            obj_ori = inputs_raw[:, 27:30]

            inputs = np.concatenate((pos_1 - base_1, pos_2 - base_2, obj_pos - pos_1, obj_pos - pos_2, obj_ori), axis=1)

            padding = np.array([[0.0 for i in range(self.state_size)] for _ in range(self.seq_length - len(inputs))])
            inputs = np.concatenate(([padding, inputs]), axis=0)

        action_gt = torch.FloatTensor(np.array(self.data_buff_actions[video_idx][frame_idx])).view(-1)
        action_gt[:3] *= 1000.0
        action_gt[7:10] *= 1000.0

        inputs = torch.FloatTensor(inputs)
        inputs = inputs.view(-1)

        vid_id = torch.FloatTensor(np.array([video_idx]))

        return inputs, action_gt, vid_id

    def __len__(self):
        return self.length