from PIL import Image

# 打开六张图片
img_a = Image.open("./acc0.3b10t/photo/acc_3b10t_plot.png")
img_b = Image.open("./acc0.3b40t/photo/acc_3b40t_plot.png")
img_c = Image.open("./acc0.3b100t/photo/acc_3b100t_plot.png")
img_d = Image.open("./acc0.7b10t/photo/acc_7b10t_plot.png")
img_e = Image.open("./acc0.7b40t/photo/acc_7b40t_plot.png")
img_f = Image.open("./acc0.7b100t/photo/acc_7b100t_plot.png")

# 获取各图片的尺寸
w_a, h_a = img_a.size
w_b, h_b = img_b.size
w_c, h_c = img_c.size
w_d, h_d = img_d.size
w_e, h_e = img_e.size
w_f, h_f = img_f.size

# 计算新图的宽度和高度
new_width = max(w_a, w_b, w_c) + max(w_d, w_e, w_f)
new_height = max(h_a, h_b, h_c) + max(h_d, h_e, h_f)



# 获取各图片的尺寸
w_a, h_a = img_a.size
w_b, h_b = img_b.size
w_c, h_c = img_c.size
w_d, h_d = img_d.size
w_e, h_e = img_e.size
w_f, h_f = img_f.size

# 计算新图的宽度和高度
new_width = max(w_a, w_b, w_c, w_d, w_e, w_f) * 3  # 三张图片横向拼接
new_height = max(h_a, h_b, h_c) + max(h_d, h_e, h_f)  # 两行图片的最大高度

# 创建一个空白新图（白色背景）
new_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

# 粘贴六张图片到新图对应位置
new_img.paste(img_a, (0, 0))  # 第一行左
new_img.paste(img_b, (w_a, 0))  # 第一行中
new_img.paste(img_c, (w_a + w_b, 0))  # 第一行右

new_img.paste(img_d, (0, h_a))  # 第二行左
new_img.paste(img_e, (w_d, h_d))  # 第二行中
new_img.paste(img_f, (w_d + w_e, h_f))  # 第二行右

# 保存新图
new_img.save("acc_6_images.jpg")

print("拼接完成，已保存为 acc_6_images.jpg")