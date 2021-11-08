import os
import sys
import pygame
import libs.envs.cfg as cfg
from libs.envs.modules import *
from itertools import chain
import time as tt
import copy
import numpy as np
import torch
from time import time
import random
from gym import spaces

class gameMap():
    def __init__(self, num_cols, num_rows):
        self.walls = []
        self.boxes = []
        self.targets = []
        self.doors = []
        self.flags = []

        self.mask_1 = False
        self.mask_2 = False

        self.hold_1 = -1
        self.hold_2 = -1

        self.over = False

        self.num_cols = num_cols
        self.num_rows = num_rows

    def addElement(self, elem_type, col, row):
        if elem_type == 'wall':
            self.walls.append(elementSprite('wall.png', col, row, cfg))
        elif elem_type == 'box':
            self.boxes.append(elementSprite('diamond.png', col, row, cfg))
        elif elem_type == 'target':
            self.targets.append(elementSprite('target.png', col, row, cfg))
        elif elem_type == 'door':
            self.doors.append(elementSprite('box.png', col, row, cfg))
        elif elem_type == 'flag':
            self.flags.append(elementSprite('flag.png', col, row, cfg))

    def draw(self, screen):
        for elem in self.elemsIter():
            elem.draw(screen)

    def elemsIter(self):
        if not self.mask_1 and not self.mask_2:
            for elem in chain(self.targets, self.walls, self.flags, self.boxes, self.doors):
                yield elem
        if self.mask_1 and not self.mask_2:
            for elem in chain(self.targets, self.walls, self.flags, self.boxes, self.doors[:1]):
                yield elem
        if not self.mask_1 and self.mask_2:
            for elem in chain(self.targets, self.walls, self.flags, self.boxes, self.doors[1:]):
                yield elem
        if self.mask_1 and self.mask_2:
            for elem in chain(self.targets, self.walls, self.flags, self.boxes):
                yield elem

    def levelCompleted(self):
        return self.over

    def constrain_action_with_block_up_down(self, blk_col, blk_row, cur_col, cur_row, del_col, del_row):
        up_limit = round(blk_row - 1.0)
        down_limit = round(blk_row + 1.0)

        if cur_row <= up_limit and (cur_row + del_row) > up_limit:
            del_row_new = round(up_limit - cur_row, 1)
        elif cur_row >= down_limit and (cur_row + del_row) < down_limit:
            del_row_new = round(down_limit - cur_row, 1)
        else:
            del_row_new = del_row

        return del_col, del_row_new, abs(del_row_new - del_row)

    def constrain_action_with_block_left_right(self, blk_col, blk_row, cur_col, cur_row, del_col, del_row):
        left_limit = round(blk_col - 1.0)
        right_limit = round(blk_col + 1.0)

        if cur_col >= right_limit and (cur_col + del_col) < right_limit:
            del_col_new = round(right_limit - cur_col, 1)
        elif cur_col <= left_limit and (cur_col + del_col) > left_limit:
            del_col_new = round(left_limit - cur_col, 1)
        else:
            del_col_new = del_col

        return del_col_new, del_row, abs(del_col_new - del_col)


    def isValidPos(self, cur_col, cur_row, del_col, del_row):

        block_size = cfg.BLOCKSIZE

        if not self.mask_1 and not self.mask_2:
            temp1 = self.walls + self.doors
        elif self.mask_1 and not self.mask_2:
            temp1 = self.walls + self.doors[:1]
        elif not self.mask_1 and self.mask_2:
            temp1 = self.walls + self.doors[1:]
        else:
            temp1 = self.walls

        count = 0

        while True:
            goal_col = round(cur_col + del_col, 1)
            goal_row = round(cur_row + del_row, 1)
            temp2 = pygame.Rect(goal_col * block_size, goal_row * block_size, block_size, block_size)
            if temp2.collidelist(temp1) == -1 or count == 10:
                break

            block_id = temp2.collidelist(temp1)
            del_col_1, del_row_1, delta_1 = self.constrain_action_with_block_up_down(temp1[block_id].col, temp1[block_id].row, cur_col, cur_row, del_col, del_row)
            del_col_2, del_row_2, delta_2 = self.constrain_action_with_block_left_right(temp1[block_id].col, temp1[block_id].row, cur_col, cur_row, del_col, del_row)

            success_1 = False
            goal_col = round(cur_col + del_col_1, 1)
            goal_row = round(cur_row + del_row_1, 1)
            temp2 = pygame.Rect(goal_col * block_size, goal_row * block_size, block_size, block_size)
            if temp2.collidelist(temp1) == -1:
                success_1 = True

            success_2 = False
            goal_col = round(cur_col + del_col_2, 1)
            goal_row = round(cur_row + del_row_2, 1)
            temp2 = pygame.Rect(goal_col * block_size, goal_row * block_size, block_size, block_size)
            if temp2.collidelist(temp1) == -1:
                success_2 = True

            if success_1 and success_2:
                if delta_1 < delta_2:
                    del_col, del_row = del_col_1, del_row_1
                else:
                    del_col, del_row = del_col_2, del_row_2
            elif success_1:
                del_col, del_row = del_col_1, del_row_1
            elif success_2:
                del_col, del_row = del_col_2, del_row_2
            else:
                del_col, del_row = del_col_2, del_row_1

            count += 1

        return del_col, del_row

    def getBox(self, col, row):
        for idd in range(len(self.boxes)):
            if abs(self.boxes[idd].col - col) < 0.15 and abs(self.boxes[idd].row - row) < 0.15:
                return idd
        return -1

    def getButton(self, col, row):
        for target in self.targets:
            if abs(target.col - col) < 0.5 and abs(target.row - row) < 0.5:
                return target
        return None

    def getFlag(self, col, row):
        for flag in self.flags:
            if abs(flag.col - col) < 0.15 and abs(flag.row - row) < 0.15:
                return flag
        return None


class gameInterface():
    def __init__(self, screen):
        self.screen = screen
        self.levels_path = cfg.LEVELDIR
        self.initGame()

    def loadLevel(self, game_level):
        with open(os.path.join(self.levels_path, game_level), 'r') as f:
            lines = f.readlines()

        self.game_map = gameMap(max([len(line) for line in lines]) - 1, len(lines))

        height = cfg.BLOCKSIZE * self.game_map.num_rows
        width = cfg.BLOCKSIZE * self.game_map.num_cols
        self.game_surface = pygame.Surface((width, height))
        self.game_surface.fill(cfg.BACKGROUNDCOLOR)
        self.game_surface_blank = self.game_surface.copy()
        for row, elems in enumerate(lines):
            for col, elem in enumerate(elems):
                if elem == 'p':
                    self.player = pusherSprite(col, row, cfg, 1)
                elif elem == 'P':
                    self.player_2 = pusherSprite(col, row, cfg, 2)
                elif elem == '*':
                    self.game_map.addElement('wall', col, row)
                elif elem == '#':
                    self.game_map.addElement('box', col, row)
                elif elem == 'o':
                    self.game_map.addElement('target', col, row)
                elif elem == 'd':
                    self.game_map.addElement('door', col, row)
                elif elem == 'f':
                    self.game_map.addElement('flag', col, row)


    def initGame(self):
        self.scroll_x = 0
        self.scroll_y = 0

    def draw(self, *elems):
        self.scroll()
        self.game_surface.blit(self.game_surface_blank, dest=(0, 0))
        for elem in elems:
            elem.draw(self.game_surface)
        self.screen.blit(self.game_surface, dest=(self.scroll_x, self.scroll_y))

    def scroll(self):
        x, y = self.player.rect.center
        width = self.game_surface.get_rect().w
        height = self.game_surface.get_rect().h
        if (x + cfg.SCREENSIZE[0] // 2) > cfg.SCREENSIZE[0]:
            if -1 * self.scroll_x + cfg.SCREENSIZE[0] < width:
                self.scroll_x -= 2
        elif (x + cfg.SCREENSIZE[0] // 2) > 0:
            if self.scroll_x < 0:
                self.scroll_x += 2
        if (y + cfg.SCREENSIZE[1] // 2) > cfg.SCREENSIZE[1]:
            if -1 * self.scroll_y + cfg.SCREENSIZE[1] < height:
                self.scroll_y -= 2
        elif (y + 250) > 0:
            if self.scroll_y < 0:
                self.scroll_y += 2



class gameEnv():
    def __init__(self, args, vis=False, current_round=0):
        if args.render_mode == 'headless':
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.args = args
        self.vis = vis
        self.current_round = current_round
        self.eval_id = 0

        pygame.display.set_caption('collaboration')
        self.screen = pygame.display.set_mode(cfg.SCREENSIZE)

        self.game_level = '1.level'
        self.dataset_name = 'dataset-continuous-info-act'
        self.round = 0

        self.count_finish = 0

        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.FloatTensor(np.array([[i, j] for i in self.linspace for j in self.linspace])).view(-1, 2)
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.game_interface = gameInterface(self.screen)
        self.game_interface.loadLevel(self.game_level)
        self.clock = pygame.time.Clock()

        self.observation_space = spaces.Box(-1.0, 1.0, (100, ))
        self.random_seed_space = spaces.Box(-1.0, 1.0, (2, ))
        self.action_space = spaces.Box(-1.0, 1.0, (4, ))

    def get_state(self):
        new_state = torch.FloatTensor(np.array([[float(self.game_interface.player.col) / 10.0, float(self.game_interface.player.row) / 10.0, float(self.game_interface.player_2.col) / 10.0, float(self.game_interface.player_2.row) / 10.0,
                float(self.game_interface.game_map.boxes[1].col) / 10.0, float(self.game_interface.game_map.boxes[1].row / 10.0),
                float(self.game_interface.game_map.boxes[0].col) / 10.0, float(self.game_interface.game_map.boxes[0].row / 10.0),
                float(self.game_interface.game_map.mask_1), float(self.game_interface.game_map.mask_2)], ]))

        self.padding = torch.cat((self.padding, new_state), dim=0)[1:]
        states = self.padding.view(1, -1)

        return states

    def update(self):
        col_1, row_1 = self.game_interface.player.col, self.game_interface.player.row
        col_2, row_2 = self.game_interface.player_2.col, self.game_interface.player_2.row

        self.game_interface.game_map.mask_1 = False
        self.game_interface.game_map.mask_2 = False

        button_1 = self.game_interface.game_map.getButton(col_1, row_1)
        if button_1:
            if button_1.col == 4 and button_1.row == 1:
                self.game_interface.game_map.mask_2 = True
            else:
                self.game_interface.game_map.mask_1 = True

        button_2 = self.game_interface.game_map.getButton(col_2, row_2)
        if button_2:
            if button_2.col == 4 and button_2.row == 1:
                self.game_interface.game_map.mask_2 = True
            else:
                self.game_interface.game_map.mask_1 = True

        self.game_interface.game_map.hold_1 = self.game_interface.game_map.getBox(col_1, row_1)
        self.game_interface.game_map.hold_2 = self.game_interface.game_map.getBox(col_2, row_2)

        flag_1 = self.game_interface.game_map.getFlag(col_1, row_1)
        flag_2 = self.game_interface.game_map.getFlag(col_2, row_2)
        flag_3 = self.game_interface.game_map.getBox(col_1, row_1)
        flag_4 = self.game_interface.game_map.getBox(col_2, row_2)

        if flag_1 and flag_2 and (flag_3 > -0.5) and (flag_4 > -0.5) and abs(col_1 - col_2) > 5.0 and abs(row_1 - row_2) > 5.0:
            self.game_interface.game_map.over = True
        else:
            self.game_interface.game_map.over = False

    def reset(self, replay=False):
        self.clock = pygame.time.Clock()
        if replay:
            self.data_file = open('{0}/{1}.txt'.format(self.dataset_name, random.randint(80, 99)), 'r')

        self.game_interface = gameInterface(self.screen)
        self.game_interface.loadLevel(self.game_level)
        self.count_finish = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise
        # print("code:", self.random_variable)

        feature_size = 10
        seq_length = 10
        self.padding = torch.FloatTensor(np.array([[0.0 for _i in range(feature_size)] for _ in range(seq_length)]))

        if self.vis:
            self.count_frames = 0
            self.eval_id += 1
            self.video_dir = 'video-results/{0}/{1}'.format(str(self.current_round), str(self.eval_id))
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)

        if replay:
            human_action_return = torch.FloatTensor([[0.0, 0.0]])
            return self.get_state(), self.random_variable, human_action_return
        else:
            return self.get_state(), self.random_variable

    def start_eval(self):
        self.eval_mode = True

    def stop_eval(self):
        self.eval_mode = False

    def step(self, actions):
        pygame.event.get()

        if self.count_finish > 300:
            success = False
            done = True

            self.reset()
            return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{'bad_transition':True}], self.random_variable
        else:
            action = actions[0].detach().cpu().numpy()
            action = np.clip(action, -0.9, 0.9)

            action[0] = round(action[0], 1)
            action[1] = round(action[1], 1)
            action[0], action[1] = self.game_interface.game_map.isValidPos(self.game_interface.player.col, self.game_interface.player.row, action[0], action[1])
            self.game_interface.player.move_delta(action[0], action[1])
            if self.game_interface.game_map.hold_1 > -0.5:
                self.game_interface.game_map.boxes[self.game_interface.game_map.hold_1].move_delta(action[0], action[1])

            action[2] = round(action[2], 1)
            action[3] = round(action[3], 1)
            action[2], action[3] = self.game_interface.game_map.isValidPos(self.game_interface.player_2.col, self.game_interface.player_2.row, action[2], action[3])
            self.game_interface.player_2.move_delta(action[2], action[3])
            if self.game_interface.game_map.hold_2 > -0.5:
                self.game_interface.game_map.boxes[self.game_interface.game_map.hold_2].move_delta(action[2], action[3])

            self.update()
            self.count_finish += 1

            self.game_interface.draw(self.game_interface.game_map, self.game_interface.player, self.game_interface.player_2)
            success = self.game_interface.game_map.levelCompleted()
            pygame.display.flip()

            if self.vis:
                pygame.image.save(self.screen, '{0}/{1}.png'.format(self.video_dir, self.count_frames))
                self.count_frames += 1

            if success:
                done = True
                self.reset()
                return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{}], self.random_variable
            else:
                done = False
                return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{}], self.random_variable

    def step_robot(self, actions):
        # print(actions)
        pygame.event.get()
        traj = self.data_file.readline()[:-1]
        traj = traj.split(' ')

        if len(traj) != 14 and self.count_finish > 30:
            success = False
            done = True
            self.data_file.close()
            self.reset(replay=True)
            human_action_return = torch.FloatTensor([[0.0, 0.0]])
            return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{'bad_transition':True}], self.random_variable, human_action_return
        else:
            action = actions[0].detach().cpu().numpy()
            action = np.clip(action, -0.9, 0.9)

            if len(traj) != 14:
                self.count_finish += 1
                action[0] = 0.0
                action[1] = 0.0
            else:
                action[0] = round(float(traj[10]), 1)
                action[1] = round(float(traj[11]), 1)
            human_action_return = torch.FloatTensor([[float(traj[10]), float(traj[11])]])
            action[0], action[1] = self.game_interface.game_map.isValidPos(self.game_interface.player.col, self.game_interface.player.row, action[0], action[1])
            self.game_interface.player.move_delta(action[0], action[1])
            if self.game_interface.game_map.hold_1 > -0.5:
                self.game_interface.game_map.boxes[self.game_interface.game_map.hold_1].move_delta(action[0], action[1])

            action[2] = round(action[2], 1)
            action[3] = round(action[3], 1)
            action[2], action[3] = self.game_interface.game_map.isValidPos(self.game_interface.player_2.col, self.game_interface.player_2.row, action[2], action[3])
            self.game_interface.player_2.move_delta(action[2], action[3])
            if self.game_interface.game_map.hold_2 > -0.5:
                self.game_interface.game_map.boxes[self.game_interface.game_map.hold_2].move_delta(action[2], action[3])

            self.update()
            self.count_finish += 1

            self.game_interface.draw(self.game_interface.game_map, self.game_interface.player, self.game_interface.player_2)
            success = self.game_interface.game_map.levelCompleted()
            pygame.display.flip()

            if self.vis:
                pygame.image.save(self.screen, '{0}/{1}.png'.format(self.video_dir, self.count_frames))
                self.count_frames += 1

            if success:
                done = True
                self.reset(replay=True)
                return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{}], self.random_variable, human_action_return
            else:
                done = False
                return self.get_state(), torch.FloatTensor([[float(success)],]), [done], [{}], self.random_variable, human_action_return


if __name__ == '__main__':
    main()
