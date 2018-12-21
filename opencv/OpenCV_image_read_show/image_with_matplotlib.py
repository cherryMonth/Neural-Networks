import cv2
from matplotlib import pyplot as plt

img = cv2.imread('target_color.png', 0)

# 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
"""
autumn	红-橙-黄
bone	黑-白，x线
cool	青-洋红
copper	黑-铜
flag	红-白-蓝-黑
gray	黑-白
hot	黑-红-黄-白
hsv	hsv颜色空间， 红-黄-绿-青-蓝-洋红-红
inferno	黑-红-黄
jet	蓝-青-黄-红
magma	黑-红-白
pink	黑-粉-白
plasma	绿-红-黄
prism	 红-黄-绿-蓝-紫-...-绿模式
spring	洋红-黄
summer	绿-黄
viridis	蓝-绿-黄
winter	蓝-绿
"""
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()
