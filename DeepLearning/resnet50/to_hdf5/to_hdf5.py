import os
import tarfile
import h5py
import imageio.v2 as imageio
from multiprocessing import Pool


def tar_to_hdf5(tar_file_name):
    with tarfile.open(tar_file_name) as tar:
        hdf5_name = tar_file_name[:-4] + '.h5'
        with h5py.File(hdf5_name, 'w') as hdf5:
            for image_name in tar.getnames():
                tar.extract(image_name)
                image = imageio.imread(image_name)
                os.remove(image_name)
                hdf5.create_dataset(image_name, data=image)
        print("{} has been built as {}".format(tar_file_name, hdf5_name))


if __name__ == "__main__":
    overall_path = r"/home/mawensen/scratch/image-net-1k/ILSVRC2012_img_train/"
    os.chdir(overall_path)

    tar_files = [file_name for file_name in os.listdir() if ".tar" in file_name]

    with Pool(8) as p:
        p.map(tar_to_hdf5, tar_files)

