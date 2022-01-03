# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:17:52 2019

@author: vincen
"""
'''利用卷积神经网络实现验证码识别'''
import os
import numpy as np
import tensorflow as tf
#导入自定义库前,将文件路径转至py文件所在路径
os.chdir('/data')
import input_data

#获取数据集路径
data_dir = '/input/captcha_pic'

if __name__ == '__main__':
    #获取训练集对象与测试集对象
    train_data,test_data = input_data.load_data(data_dir)
    #图片参数
    ImageHight = 60
    ImageWidth = 160
    ImageSize = ImageHight*ImageWidth
    #生成验证码的元素数
    OneHotNumber = 26
    IterationNumber = 100000
    
    #卷积神经网络
    with tf.name_scope('input'):
        X = tf.placeholder(shape = (None,ImageHight,ImageWidth),dtype = tf.float32)
        #每张验证码有4个字母
        y = tf.placeholder(shape = (None,4,OneHotNumber),dtype = tf.float32)
        #keep_prob
        keep_prob = tf.placeholder(tf.float32)
        #将X的shape转化为CNN需要的形式shape = (None,Height,Width,Channels)
        #因为已经将验证码转化成了灰度图，故通道数为1
        X_image = tf.reshape(X,shape = (-1,ImageHight,ImageWidth,1))
        #写入日志
#        tf.summary.image('input',X_image,max_outputs = 1)
        
    #卷积层1
    with tf.name_scope('Convolution_layer_one'):
        #设置卷积层参数
        conv1_height = 5
        conv1_width = 5
        in_channel = 1
        out_channel = 32
        conv1_strides = (1,1,1,1)
        #卷积权重
        W_conv1_shape = (conv1_height,conv1_width,in_channel,out_channel)
        W_conv1 = tf.Variable(initial_value = tf.truncated_normal(shape = W_conv1_shape,stddev = 0.01))
        b_conv1 = tf.Variable(initial_value = tf.truncated_normal(shape = [out_channel],stddev = 0.01))
        #卷积运算
        conv1 = tf.nn.conv2d(X_image,W_conv1,strides = conv1_strides,padding = 'SAME') + b_conv1
        #activation
        activation = tf.nn.relu(conv1)
        #max_pool
        #parameters
        ksize = (1,5,5,1)
        pool1_strides = (1,2,2,1)
        #池化
        pool1 = tf.nn.max_pool(activation,ksize = ksize,strides = pool1_strides,padding = 'SAME')
        dropout1 = tf.nn.dropout(pool1,keep_prob)
    
    #卷积层2
    with tf.name_scope('Convolution_layer_two'):
        conv2_height = 5
        conv2_width = 5
        in_channel = 32
        out_channel = 64
        conv2_strides = (1,1,1,1)
        #卷积权重
        W_conv2_shape = (conv2_height,conv2_width,in_channel,out_channel)
        W_conv2 = tf.Variable(initial_value = tf.truncated_normal(shape = W_conv2_shape,stddev = 0.01))
        b_conv2 = tf.Variable(tf.truncated_normal(shape = [out_channel],stddev = 0.01))
        #卷积运算
        conv2 = tf.nn.conv2d(dropout1,W_conv2,strides = conv2_strides,padding = 'SAME') + b_conv2
        #activation
        activation = tf.nn.relu(conv2)
        #max_pool
        ksize = (1,5,5,1)
        pool2_strides = (1,2,2,1)
        #池化
        pool2 = tf.nn.max_pool(activation,ksize = ksize,strides = pool2_strides,padding = 'SAME')
        dropout2 = tf.nn.dropout(pool2,keep_prob)
    
    #卷积层3
    with tf.name_scope('Convolution_layer_three'):
        conv3_height = 5
        conv3_width = 5
        in_channel = 64
        out_channel = 64
        conv3_strides = (1,1,1,1)
        #卷积权重
        W_conv3_shape = (conv3_height,conv3_width,in_channel,out_channel)
        W_conv3 = tf.Variable(initial_value = tf.truncated_normal(shape = W_conv3_shape,stddev = 0.01))
        b_conv3 = tf.Variable(tf.truncated_normal(shape = [out_channel],stddev = 0.01))
        #卷积运算
        conv3 = tf.nn.conv2d(dropout2,W_conv3,strides = conv3_strides,padding = 'SAME') + b_conv3
        #activation
        activation = tf.nn.relu(conv3)
        #max_pool
        ksize = (1,5,5,1)
        pool3_strides = (1,2,2,1)
        #池化
        pool3 = tf.nn.max_pool(activation,ksize = ksize,strides = pool3_strides,padding = 'SAME')
        dropout3 = tf.nn.dropout(pool3,keep_prob)
        
    
    
    #全连接1
    with tf.name_scope('Fully_connected_layer_one'):
        #current_image_shape = (None,上取整(ImageHeight/8),ImageWidth/8,channel) = (None,8,20,64)
        #故W.shape = (8*20*64,自己设定的值)
        W_shape = [8*20*64,1024]
        W = tf.Variable(initial_value = tf.truncated_normal(shape = W_shape,stddev = 0.01))
        b = tf.Variable(initial_value = tf.truncated_normal(shape = [1024],stddev = 0.01))
        #reshape dropout3
        dropout3 = tf.reshape(dropout3,shape = (-1,8*20*64))
        net_input = tf.matmul(dropout3,W) + b
        #activation
        activation = tf.nn.relu(net_input)
        
    #全连接2
    with tf.name_scope('Fully_connected_layer_two'):
        W_shape = [1024,4*OneHotNumber]
        W = tf.Variable(initial_value = tf.truncated_normal(shape = W_shape,stddev = 0.01))
        b = tf.Variable(initial_value = tf.truncated_normal(shape = [4*OneHotNumber],stddev = 0.01))
        net_input = tf.matmul(activation,W) + b
        #y_hat
        y_hat = tf.nn.softmax(net_input)
        
    #loss
    with tf.name_scope('loss'):
        #reshape y_hat
        y_hat_reshaped = tf.reshape(y_hat,shape = (-1,4,OneHotNumber))
        #reshape net_input
        net_input_reshaped = tf.reshape(net_input,shape = (-1,4,OneHotNumber))
        #Cross Entropy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net_input_reshaped,labels = y))
        #Adam
        grad = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cross_entropy)
#        tf.summary.histogram('Cross Entropy',cross_entropy)
    
    #accuracy:
    with tf.name_scope('accuracy'):
        predict = tf.argmax(y_hat_reshaped,axis = -1)
        true = tf.argmax(y,axis = -1)
        correct = tf.equal(predict,true)
        accuracy = tf.reduce_mean(tf.cast(correct,dtype = tf.float32))
#        tf.summary.histogram('accuracy',accuracy)
    
    #model保存器
    saver = tf.train.Saver()
    #开启会话
    with tf.Session() as sess:
#        merged = tf.summary.merge_all()
#        train_writer = tf.summary.FileWriter(r'C:\Users\vincen\Desktop',sess.graph)
#        test_writer = tf.summary.FileWriter(r'C:\Users\vincen\Desktop',sess.graph)
        accuracy_list = []
        sess.run(tf.global_variables_initializer())

        saver.save(sess,'/data/capcha_model_tmp.ckpt')
        for i in range(1,IterationNumber+1,1):
            Xbatch,ybatch = train_data.next_batch(batch_size = 800)
            _,train_accuracy = sess.run([grad,accuracy],feed_dict = {X:Xbatch,y:ybatch,keep_prob:0.8})
            #train_summary,_ = sess.run([merged,grad],feed_dict = {X:Xbatch,y:ybatch})
            #train_writer.add_summary(train_summary,i)
            if i % 10 == 0:
                X_test_batch,y_test_batch = test_data.next_batch(batch_size = 100)
                test_accuracy = sess.run(accuracy,feed_dict = {X:X_test_batch,y:y_test_batch,keep_prob:1})
                #test_summary,test_accuracy = sess.run([merged,accuracy],feed_dict = {X:X_test_batch,y:y_test_batch})
                #test_writer.add_summary(test_summary,i)
                print("第{}轮训练后,train accuracy:{:.2%},test accuracy:{:.2%}".format(i,train_accuracy,test_accuracy))
                accuracy_list.append(test_accuracy)
            if i % 1000 == 0:
                    saver.save(sess,'/data/capcha_model_tmp.ckpt')
        
#        train_writer.close()
#        test_writer.close()