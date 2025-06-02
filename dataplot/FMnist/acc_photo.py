from PIL import Image

# 打开四张图片
img_a = Image.open("./acc0.3b10t/photo/acc_3b10t_plot.png")
img_b = Image.open("./acc0.3b100t/photo/acc_3b100t_plot.png")
img_c = Image.open("./acc0.7b10t/photo/acc_7b10t_plot.png")
img_d = Image.open("./acc0.7b100t/photo/acc_7b100t_plot.png")

# 获取各图片的尺寸
w_a, h_a = img_a.size
w_b, h_b = img_b.size
w_c, h_c = img_c.size
w_d, h_d = img_d.size

# 计算新图的宽度和高度
new_width = max(w_a, w_c) + max(w_b, w_d)
new_height = max(h_a, h_b) + max(h_c, h_d)

# 创建一个空白新图（白色背景）
new_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

# 粘贴四张图片到新图对应角落
new_img.paste(img_a, (0, 0))  # 左上
new_img.paste(img_b, (new_width - w_b, 0))  # 右上
new_img.paste(img_c, (0, new_height - h_c))  # 左下
new_img.paste(img_d, (new_width - w_d, new_height - h_d))  # 右下

# 保存新图
new_img.save("acc.jpg")

print("拼接完成，已保存为 output.jpg")
