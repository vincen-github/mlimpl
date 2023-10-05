import os
import shutil

overall_path = "C:\\Users\\WenSen Ma\\OneDrive - whu.edu.cn\\桌面\\ILSVRC2012_img_val\\"
os.chdir(overall_path)

for folder in [item for item in os.listdir() if item[0] == 'n' and os.path.isdir(item)]:
    shutil.rmtree(folder)
