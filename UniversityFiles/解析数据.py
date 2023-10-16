import re
import time
import urllib.request
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

"""
提取链接数据
"""
def parse_links(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        return links

# 下载文件
def download_file(url, file_path):
    urllib.request.urlretrieve(url, file_path)

# 图片转PDF
def images_to_pdf(image_paths, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)

    for image_path in image_paths:
        c.drawImage(image_path, 0, 0, width=letter[0], height=letter[1])
        c.showPage()
    c.save()




# 用你的文件路径替换下面的路径
url = 'https://doc-e.wdd/0.png'
file_path = 'path/to/save/file.png'
download_file(url, file_path)



# 用你的文件路径替换下面的路径
file_path = 'path/to/your/file.txt'
parsed_links = parse_links(file_path)
for link in parsed_links:
    print(link)



