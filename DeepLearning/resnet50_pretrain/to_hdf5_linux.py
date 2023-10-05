import os
import tarfile
import h5py
import imageio.v2 as imageio
from datetime import datetime

overall_path = r"/home/mawensen/project/data/image-net-1k/ILSVRC2012_img_train/"
os.chdir(overall_path)

for tar_file_name in [file_name for file_name in os.listdir() if ".tar" in file_name]:
    with tarfile.open(tar_file_name) as tar:
        hdf5_name = tar_file_name[:-4] + '.h5'
        with h5py.File(hdf5_name, 'w') as hdf5:
            for image_name in tar.getnames():
                tar.extract(image_name)
                image = imageio.imread(image_name)
                os.remove(image_name)
                hdf5.create_dataset(image_name, data=image, compression="gzip", compression_opts=9)
        currentTime = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        print("{}, {} has been built as {}".format(currentTime, tar_file_name, hdf5_name))
    os.remove(tar_file_name)
