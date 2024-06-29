import pygame
import random
import time
import sys
import tkinter as tk
from tkinter import messagebox

# 加入Tkint只是为了使用messagebox
# 为了不显示Tkint的窗口，这里进行隐藏的处理
root = tk.Tk()
# 隐藏窗口
root.withdraw()
# 运行事件循环
# root.mainloop()



# 初始化pygame
pygame.init()

# 设置屏幕尺寸
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("蜘蛛移动模拟轨迹运行图")

# 定义蜘蛛类
class Spider:
    def __init__(self):
        # 添加素材
        self.image = pygame.image.load("spider.png")
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen_width - self.rect.width)
        self.rect.y = random.randint(0, screen_height - self.rect.height)
        # 横向速度 X轴
        self.speed_x = random.randint(-3, 3) 
        # 纵向速度 Y轴
        self.speed_y = random.randint(-3, 3)

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # 碰到屏幕边缘的时候进行反弹并随机改变速度方向
        if self.rect.left < 0 or self.rect.right > screen_width:
            self.speed_x = -self.speed_x
            self.speed_y = random.randint(-3, 3)
        if self.rect.top < 0 or self.rect.bottom > screen_height:
            self.speed_x = random.randint(-3, 3)
            self.speed_y = -self.speed_y

    def draw(self):
        screen.blit(self.image, self.rect)

# 加载蜘蛛图片
spider_image = pygame.image.load("spider.png")

# 创建蜘蛛列表，用于存储蜘蛛对象
spiders = []
# 游戏循环
running = True
clock = pygame.time.Clock()
start_time = time.time()

# 这里模拟每6秒增加一只蜘蛛
interval_time = 6
# 最多8只蜘蛛
max_spiders = 12

while running:
    # 控制游戏最大帧率为60
    clock.tick(60)

    # 处理游戏事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 弹出对话框询问是否结束游戏
            result = messagebox.askyesno("模型体验结束，点击[是(Y)]关闭界面", "确定要结束游戏吗？")
            if result:
                running = False

    # 更新蜘蛛位置
    for spider in spiders:
        spider.update()

    # 绘制背景为纯白色
    screen.fill((255, 255, 255))

    # 绘制蜘蛛
    for spider in spiders:
        spider.draw()

    # 新增第一只蜘蛛，这一行代码会造成满屏的蜘蛛
    # new_spider = Spider()
    # spiders.append(new_spider)

    # 是否增加新的蜘蛛
    elapsed_time = time.time() - start_time
    # 一开始就生成一只蜘蛛进行无规则的运动
    if len(spiders) == 0:
        new_spider = Spider()
        spiders.append(new_spider)

    if elapsed_time >= interval_time and len(spiders) < max_spiders:
        new_spider = Spider()
        spiders.append(new_spider)
        start_time = time.time()  # 重置计时器

    # 更新屏幕显示
    pygame.display.flip()


# 退出程序
pygame.quit()
sys.exit()
