from PIL import Image

# 打开三张图片
img1 = Image.open("./fashion.png")
img2 = Image.open("./CIFAER10.png")
img3 = Image.open("./CIFAR100.png")

# 获取各图片的尺寸
w1, h1 = img1.size
w2, h2 = img2.size
w3, h3 = img3.size

# 计算新图的宽度和高度
total_width = w1 + w2 + w3
max_height = max(h1, h2, h3)

# 创建一个空白新图（白色背景）
new_img = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

# 横向拼接三张图片
new_img.paste(img1, (0, 0))          # 左侧图片
new_img.paste(img2, (w1, 0))         # 中间图片
new_img.paste(img3, (w1 + w2, 0))    # 右侧图片

# 保存新图
new_img.save("combined_output.jpg")

print("拼接完成，已保存为 combined_output.jpg")