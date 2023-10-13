import pygame
import random
import time
import sys


# 初始化pygame
pygame.init()

# 设置屏幕尺寸
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Spider Crawler")

# 定义蜘蛛类
class Spider:
    def __init__(self):
        self.image = pygame.image.load("spider.png")  # 蜘蛛的图片
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen_width - self.rect.width)
        self.rect.y = random.randint(0, screen_height - self.rect.height)
        self.speed_x = random.randint(-3, 3)  # 蜘蛛在x轴上的速度
        self.speed_y = random.randint(-3, 3)  # 蜘蛛在y轴上的速度

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # 碰到屏幕边缘时反弹并随机改变速度方向
        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed_x = -self.speed_x
            self.speed_y = random.randint(-3, 3)
        if self.rect.top < 0 or self.rect.bottom > screen_height:
            self.speed_y = -self.speed_y
            self.speed_x = random.randint(-3, 3)

    def draw(self):
        screen.blit(self.image, self.rect)

# 加载蜘蛛图片
spider_image = pygame.image.load("spider.png")

# 创建蜘蛛列表
spiders = []

# 游戏循环
running = True
clock = pygame.time.Clock()
start_time = time.time()
spawn_time = 8  # 每8秒增加一只蜘蛛
max_spiders = 8  # 最多8只蜘蛛

while running:
    # 控制游戏最大帧率为60
    clock.tick(60)

    # 处理游戏事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 弹出对话框询问是否结束游戏
            pygame.font.init()
            font1 = pygame.font.SysFont('stxingkai', 70)
            text = font1.render("{}game over".format('my'), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (400,300)
            screen.blit(text, textRect)
            time.sleep(3)

            running = False

    # 更新蜘蛛位置
    for spider in spiders:
        spider.update()

    # 绘制背景
    screen.fill((255, 255, 255))

    # 绘制蜘蛛
    for spider in spiders:
        spider.draw()

    # 检查是否需要增加新的蜘蛛
    elapsed_time = time.time() - start_time
    if elapsed_time >= spawn_time and len(spiders) < max_spiders:
        new_spider = Spider()
        spiders.append(new_spider)
        start_time = time.time()  # 重置计时器
    if elapsed_time > 20:
        pygame.font.init()
        font1 = pygame.font.SysFont('stxingkai', 70)
        text = font1.render("{}game over".format('my'), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (400, 300)
        screen.blit(text, textRect)
        time.sleep(3)

        running = False

    # 更新屏幕显示
    pygame.display.flip()

# 退出程序
pygame.quit()
sys.exit()
