import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.utils.data as Data
from nnArchitecture import CNN

if __name__ == "__main__":

    # load data
    root = "./"
    train = datasets.MNIST(root=root,
                           train=True,
                           transform=transforms.ToTensor())
    test = datasets.MNIST(root=root,
                          train=False)

    # set main parameters
    n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.001

    # train_loader decide how to transport data
    train_loader = Data.DataLoader(dataset=train,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=2)

    # unsqueeze insert a dimension in indicated position(Here is 1)
    # Here divide 255 is not necessary.because the last layer of network architecture is softmax.
    # it implies that zoom a fixed multiple of dataset may not cause a big influence(Demonstrated by experiment.)
    test_x = torch.unsqueeze(test.data, dim=1).type(torch.FloatTensor) / 255
    test_y = test.targets

    # build cnn neural network

    cnn = CNN()
    # net architecture
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    # training and testing
    for epoch in range(n_epochs):
        for step, (x, y) in enumerate(train_loader):
            # by print x in this position. wo can find x had been zoom in [0,1)
            # print(x[0])

            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_output = cnn(test_x)
        # 1 in max(test_output, 1) indicate return the max value in each row
        pred_y = torch.max(test_output, 1)[1].squeeze()
        accuracy = sum(pred_y == test_y) / test_y.size(0)
        print("epoch: ", epoch, "| train loss: %.4f" % loss, "| test accuracy: %.4f" % accuracy)
        print(pred_y[:10], 'prediction number')
        print(test_y[:10].numpy(), 'real number')

    # save model
    path = "./model.pth"
    torch.save(cnn.state_dict(), path)
