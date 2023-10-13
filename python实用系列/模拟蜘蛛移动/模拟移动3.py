import pygame
import random
import sys
import time

# 初始化Pygame
pygame.init()

# 设置屏幕宽度和高度
screen_width = 800
screen_height = 600

# 创建屏幕对象
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Spider Game")

# 设置蜘蛛初始位置和速度
spiders = []
spider_speed = 2

# 设置计时器
spawn_timer = time.time()
spawn_interval = 8  # 秒
max_spiders = 8

# 游戏结束标志
game_over = False

# 创建一个字体对象用于显示结束信息
font = pygame.font.Font(None, 36)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if len(spiders) == max_spiders:
                game_over = True

    # 更新蜘蛛位置
    for spider in spiders:
        spider['x'] += random.randint(-spider_speed, spider_speed)
        spider['y'] += random.randint(-spider_speed, spider_speed)

    # 清空屏幕
    screen.fill((0, 0, 0))

    # 绘制蜘蛛
    for spider in spiders:
        pygame.draw.circle(screen, (255, 0, 0), (spider['x'], spider['y']), 10)

    # 更新屏幕
    pygame.display.flip()

    # 检查是否要生成新的蜘蛛
    current_time = time.time()
    if len(spiders) < max_spiders and current_time - spawn_timer >= spawn_interval:
        spiders.append({'x': random.randint(0, screen_width), 'y': random.randint(0, screen_height)})
        spawn_timer = current_time

    pygame.time.delay(100)  # 控制帧率

# 游戏结束后，显示对话框
if len(spiders) == max_spiders:
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Game Over")
    game_over_text = font.render("Game Over", True, (255, 0, 0))
    screen.blit(game_over_text, (150, 100))
    pygame.display.update()

    # 等待用户关闭窗口
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
