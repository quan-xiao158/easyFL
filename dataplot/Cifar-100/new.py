from PIL import Image

# 打开六张图片
img_a = Image.open("./acc0.3b10t/photo/acc_3b10t_plot.png")
img_b = Image.open("./acc0.3b40t/photo/acc_3b40t_plot.png")
img_c = Image.open("./acc0.3b100t/photo/acc_3b100t_plot.png")
img_d = Image.open("./acc0.7b10t/photo/acc_7b10t_plot.png")
img_e = Image.open("./acc0.7b40t/photo/acc_7b40t_plot.png")
img_f = Image.open("./acc0.7b100t/photo/acc_7b100t_plot.png")

# 计算左列和右列的最大宽度
max_left_width = max(img_a.width, img_b.width, img_c.width)
max_right_width = max(img_d.width, img_e.width, img_f.width)
total_width = max_left_width + max_right_width

# 计算每行的高度
row_heights = [
    max(img_a.height, img_d.height),
    max(img_b.height, img_e.height),
    max(img_c.height, img_f.height)
]
total_height = sum(row_heights)

# 创建新图（白色背景）
new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))

# 依次粘贴每一行的图片
y_offset = 0
# 第一行：img_a 和 img_d
new_img.paste(img_a, (0, y_offset))
new_img.paste(img_d, (max_left_width, y_offset))
y_offset += row_heights[0]

# 第二行：img_b 和 img_e
new_img.paste(img_b, (0, y_offset))
new_img.paste(img_e, (max_left_width, y_offset))
y_offset += row_heights[1]

# 第三行：img_c 和 img_f
new_img.paste(img_c, (0, y_offset))
new_img.paste(img_f, (max_left_width, y_offset))

# 保存结果
new_img.save("acc_3rows_2cols.jpg")
print("拼接完成，已保存为 acc_3rows_2cols.jpg")