# This file want to show the labels of directly generated noise image always in 2, 3, 6
# The Problem raised is that whether we can generate noise image which has another label?
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.random import randint

from TargetFunc import target_func
from nnArchitecture import CNN

# load cnn
cnn = CNN()
path = "./model.pth"
model_dict = cnn.load_state_dict(torch.load(path))

# records the fooling image labels
prob_records = []
pred_records = []
for i in range(1000):
    instance = torch.Tensor(randint(0, 256, size=(1, 28, 28)))
    # plt.imshow(instance[0], cmap=plt.cm.gray_r)
    prob, pred = target_func(instance, cnn)
    prob_records.append(prob[0])
    pred_records.append(pred[0])

counts = []
for i in range(10):
    temp = 0
    for each in pred_records:
        if each == i:
            temp += 1
    counts.append(temp)

print(counts)


sns.set()
plt.figure(dpi=400)

ax1 = plt.subplot(121)

ax1.bar(x=list(range(10)), height=counts, alpha=0.6, color="mediumpurple")
ax1.set_title('Noise Image category')
ax1.set_xlabel('digit')
ax1.set_ylabel('frequency')
ax1.set_xticks(list(range(10)))
sns.despine(right=True, top=True)

ax2 = plt.subplot(122)
ax2.hist(prob_records, alpha=0.6, color="mediumpurple")
ax2.set_title('Noise Image Confidence Predictions')
ax2.set_xlabel('confidence')
ax2.set_ylabel('frequency')
sns.despine(right=True, top=True)

plt.suptitle("Directly Generate Noise Image")
plt.show()


