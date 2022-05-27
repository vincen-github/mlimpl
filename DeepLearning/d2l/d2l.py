import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision import transforms


def use_svg_display():
    """
    Display graphs with form svg in Jupyter.Defined in :numref:`sec_calculus`
    """
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set size of graph in matplotlib.Defined in :numref:`sec_calculus
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    Set style of graph axes in matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    # set range of axes.
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """
    Plot datapoint.Defined in :numref:`sec_calculus`.
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    # Using plt.gca() can get current subplot if axes do not be pass.
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        # determine if x has only one dimension.return True if it is.
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        # expand X to two dimension if X has only one dimension.
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # clear current axes
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            # if Y is not None
            axes.plot(x, y, fmt)
        else:
            # only plot X if Y is None.
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Timer:
    """Record running time of program.Defined in :numref:`subsec_linear_model`"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """start timer"""
        self.tik = time.time()

    def stop(self):
        """stop timer and record the last running time into list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        """return cumulative time"""
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    """generate data via y = Xw + b + noise
    Defined in :numref:`sec_linear_scratch`"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def linreg(X, w, b):
    """linear model.
    Defined in :numref:`sec_linear_scratch`"""
    return X @ w + b


def squared_loss(y_hat, y):
    """mse.
    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """minibatch gradient descent.
    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a dataloader using torch.data.
    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """Returns text labels of Fashion-MNIST dataset.
    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Defined in :numref:`sec_fashion_mnist`"""
    figsize = num_cols * scale, num_rows * scale
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # Flatten axes to prevent
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # if images passed are tensor type.
            ax.imshow(img.numpy())
        else:
            # if images passed are PIL
            ax.imshow(img)
        # do not display xaxis and yaxis
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def get_dataloader_workers():
    """
    Reading data from multiple threads.
    """
    return 2


def load_data_fashion_mnist(batch_size, resize=None):
    # trans represents transforms for image dataset.The Last transform is to revert the type of images to tensor.
    trans = [transforms.ToTensor()]
    if resize:
        # if u indicate size, add it into the head of trans
        trans.insert(0, transforms.Resize(resize))
    # Compose Transforms
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def accuracy(y_hat, y):
    # calculate the number of right predict.
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # transform data type of y_hat to be y dtype and compare them.
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """
    Evaluate model accuracy given dataset.
    """
    # if the net passed is built by torch
    if isinstance(net, torch.nn.Module):
        # start evaluation mode
        # reference https://zhuanlan.zhihu.com/p/357075502
        net.eval()
    # declare an accumulator to storage the number of correct predictions and total number of predictions.
    metric = Accumulator(2)
    # clean gradient left
    with torch.no_grad():
        # Don't allow computing gradient in this context
        # reference: https://blog.csdn.net/Answer3664/article/details/99460175
        # this trick can be used to stop gradient.
        for X, y in data_iter:
            # the shape of X passed is (batch_size, 1, 28, 28)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    Train the model for an iterative cycle (definition is in chapter 3)
    """
    # if net is built by torch, switch it to training mode.
    if isinstance(net, torch.nn.Module):
        net.train()
    # metric is to record training error、training accuracy and the number of samples.
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        # if the updater is built by torch.
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # if the updater is customized.
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # return mean of training error and mean of training accuracy.
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    """drawing data in animation"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # draw multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        # if there is only single axes, self.axes will be an axes object.In else case it will be a 2D-array.
        if nrows * ncols == 1:
            # In d2l book: self.axes = [self.axes, ]
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # enable interactive mode
        plt.ion()
        # x indicates epoch, y represents (training loss, training accuracy, testing accuracy).
        # if y is scaler
        if not hasattr(y, "__len__"):
            # extend it to be a list, we can get n = 1 in this case.
            y = [y]
        n = len(y)
        # transform x shape to be same as y to set x as axis when plotting graph.
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        # Clear the current axes.
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        # self.fig.show()
        # plt.pause(0.1)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    train model in chapter 3.
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # get the sum of training error, sum of training accuracy
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        # Distinguish (test_acc,) from (test_acc).
        # addition of tuple is not equivalent to addition of corresponding elements.
        animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 5, train_loss
        assert 1 >= train_acc > 0.7, train_acc
        assert 1 >= test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """predict labels(chapter 3)"""
    X, y = iter(test_iter).next()
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
