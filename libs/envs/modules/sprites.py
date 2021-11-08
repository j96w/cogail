import os
import pygame


class pusherSprite(pygame.sprite.Sprite):
    def __init__(self, col, row, cfg, id):
        pygame.sprite.Sprite.__init__(self)
        if id == 1:
            self.image_path = os.path.join(cfg.IMAGESDIR, 'player.png')
        else:
            self.image_path = os.path.join(cfg.IMAGESDIR, 'robot.png')
        self.image = pygame.image.load(self.image_path).convert()
        color = self.image.get_at((0, 0))
        self.image.set_colorkey(color, pygame.RLEACCEL)
        self.rect = self.image.get_rect()
        self.col = col
        self.row = row

        self.speed_col = 0.7
        self.speed_row = 0.7
        self.friction = 0.1

    def update_speed(self, direction):
        if direction == 'up':
            return self.col, round(self.row - self.speed_row, 1)
        elif direction == 'down':
            return self.col, round(self.row + self.speed_row, 1)
        elif direction == 'left':
            return round(self.col - self.speed_col, 1), self.row
        elif direction == 'right':
            return round(self.col + self.speed_col, 1), self.row

    def move_delta(self, col, row):
        self.col = round(self.col + col, 1)
        self.row = round(self.row + row, 1)

    def move_to(self, col, row):
        self.col = round(col, 1)
        self.row = round(row, 1)

    def move(self, direction, is_test=False):
        if is_test:
            if direction == 'up':
                return self.col, round(self.row - self.speed_row, 1)
            elif direction == 'down':
                return self.col, round(self.row + self.speed_row, 1)
            elif direction == 'left':
                return round(self.col - self.speed_col, 1), self.row
            elif direction == 'right':
                return round(self.col + self.speed_col, 1), self.row
        else:
            if direction == 'up':
                self.row -= self.speed_row
                self.row = round(self.row, 1)
            elif direction == 'down':
                self.row += self.speed_row
                self.row = round(self.row, 1)
            elif direction == 'left':
                self.col -= self.speed_col
                self.col = round(self.col, 1)
            elif direction == 'right':
                self.col += self.speed_col
                self.col = round(self.col, 1)

    def draw(self, screen):
        self.rect.x = self.rect.width * self.col
        self.rect.y = self.rect.height * self.row
        screen.blit(self.image, self.rect)


class elementSprite(pygame.sprite.Sprite):
    def __init__(self, sprite_name, col, row, cfg):
        pygame.sprite.Sprite.__init__(self)
        self.image_path = os.path.join(cfg.IMAGESDIR, sprite_name)
        self.image = pygame.image.load(self.image_path).convert()
        color = self.image.get_at((0, 0))
        self.image.set_colorkey(color, pygame.RLEACCEL)
        self.rect = self.image.get_rect()
        self.sprite_type = sprite_name.split('.')[0]
        self.col = col
        self.row = row
        self.speed_col = 0.7
        self.speed_row = 0.7
    def draw(self, screen):
        self.rect.x = self.rect.width * self.col
        self.rect.y = self.rect.height * self.row
        screen.blit(self.image, self.rect)

    def move_delta(self, col, row):
        self.col = round(self.col + col, 1)
        self.row = round(self.row + row, 1)

    def move_to(self, col, row):
        self.col = round(col, 1)
        self.row = round(row, 1)

    def move(self, direction, is_test=False):
        if self.sprite_type == 'diamond':
            if is_test:
                if direction == 'up':
                    return self.col, round(self.row - self.speed_row, 1)
                elif direction == 'down':
                    return self.col, round(self.row + self.speed_row, 1)
                elif direction == 'left':
                    return round(self.col - self.speed_col, 1), self.row
                elif direction == 'right':
                    return round(self.col + self.speed_col, 1), self.row
            else:
                if direction == 'up':
                    self.row -= self.speed_row
                    self.row = round(self.row, 1)
                elif direction == 'down':
                    self.row += self.speed_row
                    self.row = round(self.row, 1)
                elif direction == 'left':
                    self.col -= self.speed_col
                    self.col = round(self.col, 1)
                elif direction == 'right':
                    self.col += self.speed_col
                    self.col = round(self.col, 1)