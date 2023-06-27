
import svgpathtools
from svg.path import parse_path

from svgpathtools import svg2paths
paths, attributes = svg2paths(r'C:\Users\LEGION\Desktop\2.svg')

# let's take the first path as an example
mypath = paths[0]
#mypath_1 = paths_1[0]

print("length = ", mypath.length())


# sum = 0
# for each in attributes_1:
#     print(each)
#     sum = sum + len(each)
# print(sum)


# # Find height, width
# xmin, xmax, ymin, ymax = mypath.bbox()
# print("width = ", xmax - xmin)
# print("height = ", ymax - ymin)
#
# # Let's find the area enclosed by the path (assuming the path is closed)
# try:
#     print("area = ", mypath.area())
# except AssertionError:
#     print("This path is not closed!")
#     # if you want a notion of area for an open path,
#     # you could use the bounding box area.
#     print("height*width = ", (ymax - ymin)*(xmax - xmin))
#
# d="M 27.39 23.50L27.34 23.45L27.24 23.35Q24.18 23.48 22.35 22.41L22.21 22.28L22.31 22.37Q24.82 21.00 29.69 16.78L29.58 16.66L29.64 16.72Q30.38 16.54 31.67 15.93L31.69 15.96L31.63 15.90Q30.39 21.93 30.28 28.29L30.16 28.16L30.24 28.25Q30.14 34.62 31.21 40.71L31.13 40.64L31.12 40.63Q29.55 40.01 27.42 39.89L27.27 39.75L27.44 39.92Q27.30 35.74 27.30 31.67L27.19 31.56L27.17 31.54Q27.27 27.53 27.31 23.41ZM26.83 25.56L26.93 40.25L26.98 40.29Q28.23 40.32 29.18 40.51L29.05 40.38L29.00 40.33Q29.14 41.08 29.33 42.45L29.36 42.48L29.32 42.45Q32.20 43.00 34.63 45.24L34.69 45.30L34.66 45.27Q32.22 38.18 32.18 30.72L32.16 30.70L32.11 30.65Q32.05 23.20 33.87 15.97L33.90 15.99L33.83 15.92Q33.26 16.30 31.66 17.41L31.84 17.59L31.69 17.44Q31.90 16.70 32.20 15.17L32.25 15.22L32.33 15.30Q30.90 15.97 29.46 16.35L29.55 16.44L29.47 16.36Q25.66 19.90 21.36 22.18L21.41 22.24L21.41 22.23Q22.88 23.40 25.24 23.74L25.08 23.58L25.22 23.72Q24.51 24.20 23.22 25.26L23.16 25.20L23.24 25.29Q24.66 25.60 26.87 25.60L26.85 25.58Z"
# path1 = parse_path('M 100 100 L 300 100 L 200 300 z')
#
# path2 = parse_path(d).length()
# print(len(path1))
# print(path2)

