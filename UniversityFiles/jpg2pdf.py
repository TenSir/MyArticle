from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def images_to_pdf(image_paths, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)

    for image_path in image_paths:
        c.drawImage(image_path, 0, 0, width=letter[0], height=letter[1])
        c.showPage()

    c.save()


# 文件压缩

def compress_image(input_path, output_path, quality=85):
    image = Image.open(input_path)
    image.save(output_path, optimize=True, quality=quality)

# 使用示例
input_file = 'input.jpg'
output_file = 'output.jpg'
compress_image(input_file, output_file, quality=85)


# 用你的图片路径替换下面的路径列表
image_paths = ['filefold/1.jpg', 'filefold/2.jpg', 'filefold/3.jpg']
output_path = 'output.pdf'
images_to_pdf(image_paths, output_path)
