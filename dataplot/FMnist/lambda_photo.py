from PIL import Image

# 打开两张图片
# img_left = Image.open("./KAFL/photo/lambda_KAFL_plot.png")
# img_right = Image.open("./Fedbuff/photo/lambda_FedBuff_plot.png")
img_left = Image.open("./c2fl/photo/lambda_ca2fl_plot.png")
img_right = Image.open("./fedbalance/photo/lambda_ours_plot.png")

# 获取各图片的尺寸
w_left, h_left = img_left.size
w_right, h_right = img_right.size

# 计算新图的宽度和高度（宽度相加，高度取最大值）
new_width = w_left + w_right
new_height = max(h_left, h_right)

# 创建一个空白新图（白色背景）
new_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

# 粘贴图片到左右两侧（顶部对齐）
new_img.paste(img_left, (0, 0))                # 左侧图片
new_img.paste(img_right, (w_left, 0))          # 右侧图片

# 保存新图
new_img.save("lambda_combined2.jpg")

print("拼接完成，已保存为 lambda_combined.jpg")