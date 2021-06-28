import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # ->(16, 28, 28)
            nn.ReLU(),  # ->(16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # ->(16, 14, 14)
        )  # ->(16, 14, 14)
        self.conv2 = nn.Sequential(  # ->(16, 14, 14)
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),  # ->(32, 14, 14)
            nn.ReLU(),  # ->(32, 14, 14)
            nn.MaxPool2d(kernel_size=2)  # -> (32, 7, 7)
        )
        self.out = nn.Sequential(
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of convolution layer.
        x = x.view(x.size(0), -1)
        # get distribution of label of sample.
        output = self.out(x)
        return output
