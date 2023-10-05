import os
import tarfile
import h5py
import imageio.v2 as imageio

overall_path = "C:\\Users\\WenSen Ma\\OneDrive - whu.edu.cn\\桌面\\ILSVRC2012_img_val\\"
os.chdir(overall_path)

for tar_file_name in [file_name for file_name in os.listdir() if ".tar" in file_name]:
    with tarfile.open(tar_file_name) as tar:
        hdf5_name = tar_file_name[:-4] + '.h5'
        with h5py.File(hdf5_name, 'w') as hdf5:
            for image_name in tar.getnames():
                tar.extract(image_name)
                image = imageio.imread(image_name)
                os.remove(image_name)
                hdf5.create_dataset(image_name, data=image)
        print("{} has been built as {}".format(tar_file_name, hdf5_name))