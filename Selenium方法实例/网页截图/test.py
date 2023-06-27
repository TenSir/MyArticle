
import os
path = r'C:\Users\LEGION\Downloads'
import os

# print(os.path.abspath(path))

for filename in os.listdir(path):
    file = os.path.abspath(path) + '\\' + filename
    print(file)