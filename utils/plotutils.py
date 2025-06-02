from PIL import Image

def horizontal_concat(image1_path, image2_path, output_path):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 确保图片模式一致（如RGB）
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')

    # 获取单张图片尺寸
    width, height = img1.size

    # 创建新画布（宽度双倍，高度不变）
    new_image = Image.new('RGB', (width * 2, height))

    # 拼接图片
    new_image.paste(img1, (0, 0))         # 第一张图在左侧
    new_image.paste(img2, (width, 0))     # 第二张图在右侧

    # 保存结果
    new_image.save(output_path)

# 使用示例
horizontal_concat('image1.png', 'image2.png', 'combined.jpg')