import os
import sys
import pygame


def Button(screen, position, text, cfg):
    bwidth = 310
    bheight = 65
    left, top = position
    pygame.draw.line(screen, (150, 150, 150), (left, top), (left + bwidth, top), 5)
    pygame.draw.line(screen, (150, 150, 150), (left, top - 2), (left, top + bheight), 5)
    pygame.draw.line(screen, (50, 50, 50), (left, top + bheight), (left + bwidth, top + bheight), 5)
    pygame.draw.line(screen, (50, 50, 50), (left + bwidth, top + bheight), [left + bwidth, top], 5)
    pygame.draw.rect(screen, (100, 100, 100), (left, top, bwidth, bheight))
    font_path = os.path.join(cfg.FONTDIR, 'simkai.ttf')
    font = pygame.font.Font(font_path, 50)
    text_render = font.render(text, 1, (255, 0, 0))
    return screen.blit(text_render, (left + 50, top + 10))


def startInterface(screen, cfg):
    screen.fill(cfg.BACKGROUNDCOLOR)
    clock = pygame.time.Clock()
    while True:
        button_1 = Button(screen, (95, 150), 'start', cfg)
        button_2 = Button(screen, (95, 305), 'exit', cfg)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_1.collidepoint(pygame.mouse.get_pos()):
                    return
                elif button_2.collidepoint(pygame.mouse.get_pos()):
                    pygame.quit()
                    sys.exit(0)
        clock.tick(60)
        pygame.display.update()


def endInterface(screen, cfg):
    screen.fill(cfg.BACKGROUNDCOLOR)
    clock = pygame.time.Clock()
    font_path = os.path.join(cfg.FONTDIR, 'simkai.ttf')
    text = 'Congrats!'
    font = pygame.font.Font(font_path, 30)
    text_render = font.render(text, 1, (255, 255, 255))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.blit(text_render, (120, 200))
        clock.tick(60)
        pygame.display.update()

def switchInterface(screen, cfg):
    screen.fill(cfg.BACKGROUNDCOLOR)
    clock = pygame.time.Clock()
    while True:
        button_1 = Button(screen, (95, 150), 'next', cfg)
        button_2 = Button(screen, (95, 305), 'exit', cfg)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_1.collidepoint(pygame.mouse.get_pos()):
                    return
                elif button_2.collidepoint(pygame.mouse.get_pos()):
                    pygame.quit()
                    sys.exit(0)
        clock.tick(60)
        pygame.display.update()