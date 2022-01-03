# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:36:56 2019

@author: vincen
"""
#Gan : 对抗生成网络
'''
    model:
        D:discriminator
            input:
                x:true image
                z:fake image
            D_loss:
                -log(D(x))-log(1-D(G(z)))
            output:
                scaler in [0,1]
        G:generator
            input:
                random vector
            G_loss:
                -log(D(G(z)))
            output:
                z:fake image
            
        target:minnimize G_loss and D_loss
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns

#reset the graph of tensorflow
tf.reset_default_graph()

# generator
def generator(noise,g_units,out_dim,alpha):
    '''
    parameters:
        g_units:
            the number of neurons in hidden layer
        out_dim:
            the dimension of output 
        alpha:
            the parameters of leaky relu
    '''
    with tf.variable_scope('generator'):
        #hidden_layer1
        hidden_layer1 = tf.layers.dense(noise,g_units)
        #activation function relu
    #    activation = tf.nn.relu(hidden_layer1)
        #leaky_relu
        activation1 = tf.maximum(alpha*hidden_layer1,hidden_layer1)
        #drop_out
        #rate  = 1 - keep_prob
        dropout = tf.nn.dropout(activation1,rate = 0.2)
        #output
        hidden_layer2 = tf.layers.dense(dropout,out_dim)
        #activation2 
        activation2 = tf.nn.tanh(hidden_layer2)
        return activation2
    
#discriminator
def discriminator(image,d_units,alpha,reuse = False):
    '''
    parameters:
        d_units:
            the number of neurons in hidden layer
        reuse:
            the parameters of tensorflow
            because we need to train discriminator twice on one epoch
            so we need to set the reuse parameters to true when training the discriminator in the second time on one iteration
    '''
    with tf.variable_scope('discriminator',reuse =  reuse):
#        hidden_layer1
        hidden_layer1 = tf.layers.dense(image,d_units)
#        activation1
        activation1 = tf.maximum(alpha*hidden_layer1,hidden_layer1)
#        hidden_layer2
        logits = tf.layers.dense(activation1,1)
#        activation2 sigmoid
        activation2 = tf.nn.sigmoid(logits)
        return logits,activation2
    

        
if __name__ == '__main__':
    '''set parameters'''
    #read mnist data
    os.chdir(r'C:\Users\vincen\Desktop\资料\Machine Learning\data')
    mnist = input_data.read_data_sets('Mnist')
#    the size of noise data for generator
    noise_size = 100
#    the size of real image
    image_size = mnist.train.images[0].shape[0]
#    the number of generator's neurons
    g_units = 128
#    the number of discriminator's neurons
    d_units = 128
    #the parameters of generator's first activation leaky relu
    alpha = 0.1
    learning_rate = 0.0001
    
    #build network:
    real_image = tf.placeholder(dtype = tf.float32,shape = [None,image_size])
    noise = tf.placeholder(dtype = tf.float32,shape = [None,noise_size])
    #get fake image
    G_image = generator(noise,g_units,image_size,alpha = alpha)
    #G image score
    fake_logits,fake_score = discriminator(G_image,d_units,alpha)
    #real image_score
    real_logits,real_score = discriminator(real_image,d_units,alpha,reuse = True)
    #build loss
    #Generator loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits,labels = tf.ones_like(real_logits)))
    #Discriminator loss
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logits,labels = tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits,labels = tf.zeros_like(real_logits)))
    D_loss = tf.add(real_loss,fake_loss)
    
    #varlist
    train_var = tf.trainable_variables()
    G_var = [var for var in train_var if var.name.startswith('generator')]
    D_var = [var for var in train_var if var.name.startswith('discriminator')]
    
    #optimizer
    d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss,var_list = D_var)
    g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_loss,var_list  = G_var)
    
    #Saver
    saver = tf.train.Saver()
    
    #train
    batch_size = 128
    #the number of iterations
    epoch = 1000
    #save the instance
    instance = []
    #g_loss_list
    g_loss_list = []
    #d_loss_list
    d_loss_list = []
    #session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in range(mnist.train.num_examples//batch_size):
    #            get real image
                batch_image,_ = mnist.train.next_batch(batch_size)
                batch_image = 2*batch_image - 1
    #            get noise
                noise_image = np.random.uniform(-1,1,size = [batch_size,noise_size]).astype(np.float32)
    #            trian
                sess.run(g_train_opt,feed_dict = {noise:noise_image,real_image:batch_image})
                sess.run(d_train_opt,feed_dict = {noise:noise_image,real_image:batch_image})
                #partical loss
                d_loss = sess.run(D_loss,feed_dict = {noise:noise_image,real_image:batch_image})
                g_loss = sess.run(G_loss,feed_dict = {noise:noise_image,real_image:batch_image})
            print('epoch:{},d_loss:{},g_loss:{}'.format(i,d_loss,g_loss))
            #save result
            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)
            if i%100 == 0:
                for j in range(8):
                    fake_image = (sess.run(G_image,feed_dict = {noise:noise_image})[j] + 1)/2
                    instance.append(fake_image)
            #
#            fake_image1 = (sess.run(G_image,feed_dict = {noise:noise_image})[0] + 1)/2
#            plt.figure()
#            plt.imshow(fake_image1.reshape(28,-1),cmap = 'Greys_r')
#            plt.show()
            '''saver model'''
#        saver.save(sess,r'C:\Users\vincen\Desktop\model')
            
            
    '''image grid'''
    plt.rcParams['font.sans-serif'] = ['FangSong'] 
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(dpi = 300)
    for i in range(80):
        #调整子图间距
        plt.subplots_adjust(left = 0.15,bottom = 0.1,top = 0.9,right = 0.95,hspace = 0,wspace = 0)
        #fig.tight_layout()
        #plt.subplots_adjust(wspace = 0,hspace = 0)
        plt.subplot(10,8,i+1)
        plt.imshow(instance[i].reshape(-1,28),cmap = 'Greys_r')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle('对抗生成网络')
    plt.show()
    
    '''loss'''
    total_loss = []
    for i in range(1000):
        total_loss.append(d_loss_list[i]+g_loss_list[i])
    sns.set()
    plt.figure(dpi = 110)
    x = np.linspace(1,1000,1000)
    plt.plot(x,d_loss_list,label = 'Dloss',color = 'b')
    plt.plot(x,g_loss_list,label = 'Gloss',color = 'g')
    plt.plot(x,total_loss,label = 'Total loss',color = 'r')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss_value')
    plt.title('Gan loss figure')
    plt.show()
    
    
        

        
        
            
    
    
    
    