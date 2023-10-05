import os
import tarfile

overall_path = "C:\\Users\\WenSen Ma\\OneDrive - whu.edu.cn\\桌面\\ILSVRC2012_img_val\\"
os.chdir(overall_path)
for folder in [item for item in os.listdir() if item[0] == 'n' and os.path.isdir(item)]:
    with tarfile.open(overall_path + folder + '.tar', "w") as tar:
        os.chdir(overall_path + folder)
        for file in os.listdir():
            tar.add(file)
