from d2l import nn, data, synthetic_data, load_array, torch


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor(4.2)
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))

    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss(reduction='mean')

    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            # 训练过程通常使用mini-batch方法，若不将梯度清零，梯度会在第二次backward时在上一次计算出梯度的基础上进行累加。
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print("epoch {}, loss {}".format(epoch + 1, l))

        print("***", net[0].weight.grad)
        print(dir(net[0]))
        print(dir(net[0].weight))
        w = net[0].weight.data
        print("error of w: {}".format(true_w - w.reshape(true_w.shape)))
        b = net[0].bias.data
        print('error of b: {}'.format(true_b - b))
