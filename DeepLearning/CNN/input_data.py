# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 19:16:36 2019

@author: vincen
"""
import os
from PIL import Image
import numpy as np
def load_data(data_dir):
    '''
    ----------------------------
    parameters:
        data_dir:str
            数据集所在文件的上一路径
    ----------------------------
    Return:
        dtype:DataSet object
        训练集和测试集
    '''
    #通过传入参数确定训练集与测试集文件所在路径
    train_dir = os.path.join(data_dir,'train')
    test_dir = os.path.join(data_dir,'test')
    #调用read_images_and_labels()方法分别获取训练集与测试集,再将其转化为DataSet object
    train_images,train_labels = read_images_and_labels(train_dir)
    test_images,test_labels = read_images_and_labels(test_dir)
    return  DataSet(train_images,train_labels),DataSet(test_images,test_labels)

def read_images_and_labels(path):
    '''
    读取图片与对应标签队列的方法
    
    ---------------------------
    parameters:
        path:str
            需要读取数据所在目录
    --------------------------
    Return:
        ndarray
            特征及对应标签
    '''
    #进入传入的目录
    os.chdir(path)
    images,labels = [],[]
    for filename in os.listdir(path):
        #通过下面定义的read_image()与read_labels()方法获取到目标特征与标签
        images.append(read_image(filename))
        labels.append(read_label(filename))
    return np.array(images),np.array(labels)

def read_image(filename):
    '''
    通过文件名获取图片信息的方法
    
    --------------------------------
    parameters:
        filename:str
            图片文件名
    ---------------------------------
    Return:
        ndarray
            单张图片对应的ndarray数组(灰度图)
            shape = (40,160)
    '''
    #Image.open()返回pillow库的image对象
    #convert('L')转化为灰度图
    image = Image.open(filename).convert('L')
    image_data = np.asarray(image)
    return image_data


def read_label(filename):
    '''
    获取图像标签(文件名)的方法,内部包含了OneHotEncoding
    
    --------------------------------
    parameters:
        filename:str
            图片文件名
    -------------------------------
    Return:
        单张图片对应的标签(OneHotEncoding)
        shape = (4,26)
    '''
    #去除后缀
    label = filename.split('.')[0]
    label_ohe = []
    for letter in label:
        #获取图片名称中每一位字母的位置
        letter_index = ord(letter) - ord('A')
        #OneHotEncoding
        #共有26个大写字母
        temp = [0]*26
        temp[letter_index] = 1
        label_ohe.append(temp)
    return np.asarray(label_ohe)

#定义数据集类
class DataSet(object):
    '''自定义数据集类'''
    
    def __init__(self,images,labels):
        '''
        初始化方法
        
        --------------------------------------------
        parameters:
            images:ndarray
                生成数据集的多张图片的信息
            labels:ndarray
                生成数据集的多张图片的OneHotEncoding标签
        -----------------------------------------------
        '''
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        #遍历完了几个数据集
        self._epochs_completed = 0
        #记录数据集中的位置
        self._index_in_epoch = 0
    
    def images(self):
        return self._images
    
    def labels(self):
        return self._labels
    
    def epochs_completed(self):
        return self._epochs_completed
    
    def _index_in_epoch(self):
        return self._index_in_epoch
    
    def next_batch(self,batch_size):
        '''
            返回下一个批次数据的方法
        
        -------------------------------
        parameters:
            batch_size:int
                批次大小
        -------------------------------
        Return:
            dtype = tuple
            shape = (样本特征，样本标签)
        '''
        #若批次大小大于数据集数量大小，报错
        assert batch_size <= self._num_examples
        
        #若next_batch()后超出数据集大小,则回到数据集起始位置
        if batch_size + self._index_in_epoch > self._num_examples:
            #遍历数据集数量+1
            self._epochs_completed += 1
            #游标位置归0
            self._index_in_epoch = 0
            
        #若新批次开始,进行数据集的洗牌操作
        if self._index_in_epoch == 0:
            #生成新的索引
            perm = np.arange(self._num_examples)
            #打乱
            np.random.shuffle(perm)
            #获取新的数据集
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        return self._images[start:self._index_in_epoch],self._labels[start:self._index_in_epoch]
    
if __name__ == '__main__':
    data_dir = r'C:\Users\vincen\Desktop\captcha_pic'
    train_data,test_data = load_data(data_dir)
    train_images_batch,train_labels_batch = train_data.next_batch(batch_size = 100)
    single_image = train_images_batch[0]
    single_label = train_labels_batch[0]
    #转化这个label
    label_temp = np.argmax(single_label,axis = -1)
    label = ''.join([chr(num + ord('A')) for num in label_temp])
    print('label:',label)
    #可视化,测试是否对应
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(single_image,cmap = 'gray')
    plt.axis('off')
    plt.show()
    
    















