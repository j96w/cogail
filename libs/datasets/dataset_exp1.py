import numpy as np
import torch
import torch.utils.data


class Game_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.data_ids = [i for i in range(0, 40)] + [i for i in range(100, 140)] + [i for i in range(150, 190)] # dataset 20-20-40-40

        self.data_buff = {}
        self.video_idx_list = []
        self.frame_idx_list = []

        self.seq_length = 10

        self.dataset_name = self.args.gail_experts_dir

        for item in self.data_ids:
            f = open('{0}/{1}.txt'.format(self.dataset_name, item), 'r')
            prior_str = f.readline()[:-1]
            tmp = prior_str.split(' ')
            self.data_buff[item] = [[float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11]), float(tmp[12]), float(tmp[13])], ]
            while True:
                tmp_str = f.readline()[:-1]
                if tmp_str == '':
                    break
                elif prior_str == tmp_str:
                    continue
                else:
                    prior_str = tmp_str
                    tmp = tmp_str.split(' ')
                    self.data_buff[item].append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11]), float(tmp[12]), float(tmp[13])])

            # print(self.data_buff[item])

            self.video_idx_list += [item for i in range(len(self.data_buff[item]) - 1)]
            self.frame_idx_list += [i for i in range(len(self.data_buff[item]) - 1)]

        self.length = len(self.video_idx_list)

        print(self.length, self.length, self.length, self.length)

    def __getitem__(self, index):

        video_idx = self.video_idx_list[index]
        frame_idx = self.frame_idx_list[index]

        if frame_idx >= (self.seq_length - 1):
            inputs = np.array(self.data_buff[video_idx][(frame_idx + 1 - self.seq_length):(frame_idx + 1)])[:, :-4]
        else:
            inputs = np.array(self.data_buff[video_idx][:(frame_idx + 1)])[:, :-4]
            padding = np.array([[0.0 for _i in range(10)] for _ in range(self.seq_length - len(inputs))])
            inputs = np.concatenate(([padding, inputs]), axis=0)

        if (frame_idx + 1 + self.seq_length) < len(self.data_buff[video_idx]):
            action_human_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-4:-2]).view(-1)
            action_robot_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-2:]).view(-1)
            action_gt = torch.cat((action_human_gt, action_robot_gt), dim=0)
        else:
            action_human_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-4:-2]).view(-1)
            action_robot_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-2:]).view(-1)
            action_gt = torch.cat((action_human_gt, action_robot_gt), dim=0)

        inputs = torch.FloatTensor(inputs)
        inputs[:, :8] /= 10.0
        inputs = inputs.view(-1)

        vid_id = torch.FloatTensor(np.array([video_idx]))

        return inputs, action_gt, vid_id

    def __len__(self):
        return self.length