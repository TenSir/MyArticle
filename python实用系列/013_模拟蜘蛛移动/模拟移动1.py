import pygame
import random
import sys

# 初始化Pygame
pygame.init()

# 设置屏幕宽度和高度
screen_width = 800
screen_height = 600

# 创建屏幕对象
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Spider Movement")

# 设置蜘蛛初始位置
spiders = [{'x': random.randint(0, screen_width), 'y': random.randint(0, screen_height)}]

# 设置计时器
spawn_timer = 8 * 1000  # 毫秒
spawn_interval = 8 * 1000  # 毫秒
end_simulation = False

# 蜘蛛速度
spider_speed = 2

# 设置字体和文本
font = pygame.font.Font(None, 36)
end_text = font.render("Simulation Ended", True, (255, 255, 255))
end_text_rect = end_text.get_rect(center=(screen_width // 2, screen_height // 2))

while not end_simulation:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            end_simulation = True

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
    current_time = pygame.time.get_ticks()
    if current_time - spawn_timer >= spawn_interval:
        spiders.append({'x': random.randint(0, screen_width), 'y': random.randint(0, screen_height)})
        spawn_timer = current_time

    # 如果时间达到50秒，显示结束文本
    if current_time >= 50000:
        screen.blit(end_text, end_text_rect)
        pygame.display.update()

    pygame.time.delay(100)  # 控制帧率

# 游戏循环结束后，等待用户关闭窗口
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
