import torch
from matplotlib import pyplot as plt
from numpy import concatenate, argmax, mean, asarray
from numpy.random import uniform, randint
from torch import Tensor

from nnArchitecture import CNN

# in this part. i will use evolution algorithm to generate adversarial examples which target is not in 2,3,6


# load nn model
cnn = CNN()
path = "./model.pth"
model_dict = cnn.load_state_dict(torch.load(path))


def target_func(target_label, x: Tensor, nn=cnn) -> float:
    # transform shape of x to (1, 1, 28, 28) and its type to Tensor
    x = torch.Tensor(x)
    # the second 1 represent the channel, 1 is the position u want to add to.
    return nn(x.unsqueeze(1))[0][target_label].detach().numpy()


class EA(object):
    """===== implement evolutionary algorithm by vincen====="""

    def __init__(self, target_label, target_func=target_func, cross_prob=0.2,
                 cross_len=15, mutation_len=15, mutation_prob=0.3, init_pops=50,
                 evolution_nums=60, out_rate=0.27):
        # target_label represents the category of adversarial sample to be generated
        self.target_label = target_label
        # target function  inout: image and neural network -> target
        self.target_func = target_func
        # Probability of cross
        self.cross_prob = cross_prob
        # Length of cross
        self.cross_len = cross_len
        # Probability of mutation
        self.mutation_prob = mutation_prob
        # Length of mutation
        self.mutation_len = mutation_len
        # initial number of populations
        self.init_pops = init_pops
        # number of evolutions
        self.evolution_nums = evolution_nums
        # out_rate represents the percentage of individuals that are not eliminated in each round to the total
        self.out_rate = out_rate

    def cross(self, pops):
        for i in range(len(pops)):
            # decide whether do cross operate to current instance
            u = uniform(0, 1)
            if u < self.cross_prob:
                # decide which individual crosses the current instance
                index = randint(0, len(pops))
                # decide starting position of cross operation
                start_pos = randint(0, 28 * 28)
                end_pos = start_pos + self.cross_len
                # Crossover to produce new individuals
                new1 = concatenate((pops[index][:start_pos], pops[i][start_pos:end_pos]))
                new1 = concatenate((new1, pops[index][end_pos:]))
                pops.append(new1)

                new2 = concatenate((pops[i][:start_pos], pops[index][start_pos:end_pos]))
                new2 = concatenate((new2, pops[i][end_pos:]))
                pops.append(new2)

        return pops

    def mutate(self, pops):
        for i in range(len(pops)):
            # decide whether do mutation operate to current instance
            u = uniform(0, 1)
            if u < self.mutation_prob:
                # decide starting position of cross operation.
                # note that start position can not overout 28*28 - self.mutation_len
                start_pos = randint(0, 28 * 28 - self.mutation_len)
                end_pos = start_pos + self.mutation_len
                pops[i][start_pos:end_pos] = uniform(0, 1, self.mutation_len)

        return pops

    def select(self):
        """Return: the index of instance selected."""
        # select instance on pops
        # calculate the probability of each instance.
        select_prob = self.fitness / sum(self.fitness)
        # cumulative probability
        cum_prob = [sum(select_prob[: i + 1]) for i in range(len(select_prob))]

        u = uniform(0, 1)
        for i in range(len(cum_prob)):
            if u < cum_prob[i]:
                return i

    def evolute(self):
        # t represents time
        t = 0
        # pops is used to storage population
        pops = []
        # initialize population
        for i in range(self.init_pops):
            # u only need to generate the images whose pixel is in range (0, 1] because cnn architecture.
            # please note that the instance in pops is a vector which length 28 * 28, u must be reshape it to a square
            # when u take it into target_func
            pops.append(uniform(0, 1, size=28 * 28))

        # Evaluate the fitness of each individual through the objective function
        self.fitness = [self.target_func(self.target_label, x=each.reshape(1, 28, 28)) for each in pops]

        # start evolution
        while t < self.evolution_nums:
            # if The population is extinct, withdraw from optimization.
            if len(self.fitness) == 0:
                return pops, self.fitness
            # If the average fitness of the population is greater than 0.9,
            # it means that the goal has been achieved and the optimization is exited.
            if mean(self.fitness) > 0.75:
                return pops, self.fitness
            print("target_label:{}, time:{}, max_fitness:{}, mean_fitness:{}".format(self.target_label, t,
                                                                                     max(self.fitness),
                                                                                     mean(self.fitness)))
            # starting to select.reserve (1 - self.out_rate) * len(pops) instance
            temp_pops = []
            for i in range(int(len(pops) * (1 - self.out_rate))):
                selected_index = self.select()
                # remove sample selected in fitness
                self.fitness.pop(selected_index)
                # add sample selected to selected_pops
                temp_pops.append(pops.pop(selected_index))
            # cross
            temp_pops = self.cross(temp_pops)
            # mutation
            pops = self.mutate(temp_pops)
            # Recalculate fitness
            # (because a new individual is born, it is guaranteed that fitness and pops can correspond when returning)
            self.fitness = [self.target_func(self.target_label, x=each.reshape(1, 28, 28)) for each in pops]

            t += 1

        return pops, self.fitness


# （2）个体评价：计算群体P(t)中各个个体的适应度。 [2]
# （3）选择运算：将选择算子作用于群体。选择的目的是把优化的个体直接遗传到下一代或通过配对交叉产生新的个体再遗传到下一代。选择操作是建立在群体中个体的适应度评估基础上的。 [2]
# （4）交叉运算：将交叉算子作用于群体。遗传算法中起核心作用的就是交叉算子。 [2]
# （5）变异运算：将变异算子作用于群体。即是对群体中的个体串的某些基因座上的基因值作变动。群体P(t)经过选择、交叉、变异运算之后得到下一代群体P(t+1)。 [2]
# （6）终止条件判断：若t=T,则以进化过程中所得到的具有最大适应度个体作为最优解输出，终止计算。 [2]
# 遗传操作包括以下三个基本遗传算子(genetic operator):选择(selection)；交叉(crossover)；变异(mutation)。 [1]


if __name__ == "__main__":

    fig = plt.figure(dpi=600)

    for label in range(10):
        # creat instance of ea
        ea = EA(target_label=label, target_func=target_func)

        while True:
            flag = False
            pops, fitness = ea.evolute()
            fitness = asarray(fitness)
            # counter is to count number of images in each column
            counter = 0
            for i in range(len(pops)):
                if fitness[i] > .95:
                    # plot
                    ax = plt.subplot(5, 10, counter * 10 + label + 1)
                    ax.imshow(pops[i].reshape(28, 28), cmap=plt.cm.gray_r)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    counter += 1
                    # if we has plot 5 images.break inner loop and outer loop
                    if counter == 5:
                        flag = True
                        break
            # if flag == False, it implies that the number of fooling image in this column is not fill with
            # u should generate another images satisfy condition through ea
            if flag:
                break
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # 调整子图间距
    plt.show()
