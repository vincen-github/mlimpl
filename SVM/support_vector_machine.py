# %%
import numpy as np

class SVM(object):
    """
    SVM
    =====
    @author: vincen

    @email: vincen.nwu.@gmail.com

    @github: https://github.com/vincen-github/Machine-Learning-Code

    Reference : 
    
            1.《统计学习方法》—— 李航

            2. Sequential Minimal Optimization:A Fast Algorithm for Training Support Vector Machines——John C.Platt 1998

    Available attributes
    ---------------------

        1. kernel: str, defalut = 'rbf'
            It must be one of 'linear', 'rbf' or a callable.If none is given, 'rbf' will be used.

        2. C: float, default = 1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C.You must set this parameter to be greater than or equal to 0.

        3. tol : float, default = 1e-3
            Tolerance for stopping criterion.
        
        4. eps : float, default = 1e-3
            when eps is negative.The parameters is valid.

    Available function
    ---------------------
        1. fit(self, X, y):
            Fit the SVM model using SMO algorithm according to the given training data.

            Parameters
            ----------
            X : ndarray
                Feature matrix
            y : ndarray
                Target vector(class labels in classification must be binary)

        2. kernel_(self, x1, x2, gamma = 'auto'):
            Using kernel function to calcuate the inner product in high dimension space.
            if none is given.uses 1 / nfeatures.

            Parameters
            ---------
            x1 : ndarray.  
                The first coordinate of the feature space(before kernel mapping).
            x2 : ndarray 
                The second coordinate of the feature space.
            gamma : float, default = 'auto'
                the parameters in rbf kernel function.The form of this function is as follows
                    K(x1, x2) = exp(-γ||x1 - x2||²)
                    notation: If use linear kernel function.You can ignore this parameters.
        
        3. takeStep(self, i1, i2):
            Update the alpha1, alpha2, w, b after determining the two optimized Lagrangian multipliers.
            
            Parameters:
            ------------
                i1: Index of the first  Lagrangian multiplier.
                i2: Index of the second Lagrangian multiplier.

        4. examineExample(self):
            Determine alpha1 and alpha2 in each round of optimization.

            Determination of alpha2
            ---------------------
            If an example violates the KKT conditions.it is then eligible for optimization.

            Determination of alpha1
            ----------------------
            maximize the |E1 - E2|
            If E1 is positive, SMO chooses an example with minimum error E2.If E1 is negative, SMO chooses an example with maximum error E2.
        
        5. __updateEi(self):
            private function.Used to update the error of each round.
        
        5. predict_single_sample(self, x):
            predict the object function value according to inputing vector x.
            It should be noted that sign(object function value) = target.

        6. predcit(self, X):
            predict the target according to multiple smaples at the same time.

        7. score(self, X, y):
                score = ΣI{y == y_pred}/X.shape[0]
    """
    def __init__(self, kernel = 'rbf', C = 1, tol = 1e-3, eps = 1e-3):
        if kernel not in ["linear", 'rbf']:
            raise Exception("Invalid kernel function.")
        if C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        # 初始化惩罚因子
        self.C = C
        self.tol = tol
        self.eps = eps
        # 初始化核函数
        self.kernel = kernel
        # 初始化超平面的bias为0
        self.b = 0

    def kernel_(self, x1, x2, gamma = 'auto'):
        if self.kernel == 'linear':
            return x1.T@x2
        
        if self.kernel == 'rbf':
            if gamma == 'auto':
                gamma = 1 / self.ncol
            if not isinstance(gamma, (float, int)):
                raise TypeError("gamma must be float or int;got(%s)" % type(gamma))
            if gamma <= 0:
                raise ValueError("gamma must be positive; got (gamma = %r)" % gamma)

            return np.exp(-gamma*np.power(np.linalg.norm(x1 - x2), 2))
    
    def takeStep(self, i1, i2):
        # 若选出的两个乘子的Index相同,此时优化失败。
        if i1 == i2:
            return 0
        # 通过i1, i2索引需要的alpha,E,y
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.Ei[i1]
        E2 = self.Ei[i2]
        s = y1*y2
        # 计算修剪上下限
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        elif y1 == y2:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        # 当L == H,alpha2可取的值确定,此时没有优化空间了,判定为优化失败.
        if L == H:
            return 0
        # 计算kernel value
        K11 = self.kernel_(self.X[i1], self.X[i1])
        K22 = self.kernel_(self.X[i2], self.X[i2])
        K12 = self.kernel_(self.X[i1], self.X[i2])
        # 计算alpha2更新公式中的eta
        eta = K11 + K22 - 2*K12
        
        if eta > 0:
            a2 = alpha2 + y2*(E1 - E2)/eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H

        #若eta ≤ 0,W(alpha2)的二阶导数 W"(alpha2) = eta < 0,
        # 即W(alpha2)作为二次函数开口向上,此时在边界L或者H取得最小值
        # 此时需要计算目标函数值在L和H的函数值,取较小的函数值对应的边界作为alpha2的更新值。
        # 下面是关于L,H处函数值的推导
        #-------------------------------------------------------
        # Reference: https://zhuanlan.zhihu.com/p/336579565
        # --------------------------------------------------------
        else:
            f1 = y1*(E1 - self.b) - alpha1*K11 - s*alpha2*K12
            f2 = y2*(E2 - self.b) - alpha2*K22 - s*alpha1*K12
            L1 = alpha1 + s*(alpha2 - L)
            H1 = alpha2 + s*(alpha2 - H)
            Lobj = L1*f1 + L*f2 + (1/2)*(L1**2)*K11 + (1/2)*(L**2)*K22 + s*L*L1*K12 - (L + L1)
            Hobj = H1*f1 + H*f2 + (1/2)*(H1**2)*K11 + (1/2)*(H**2)*K22 + s*H*H1*K12 - (H + H1)

            if Lobj < Hobj - self.eps:
            # a2暂时存储新的alpha2的值(因为后面还要用到alpha2_old)
                a2 = L
            elif Lobj > Hobj + self.eps:
                a2 = H
            # 若两边函数值相差不大,则会在下面判定更新失败
            else:
                a2 = alpha2
        if abs(a2 - alpha2) < self.eps*(a2 + alpha2 + self.eps):
            return 0
        # 若更新成功,利用公式更新alpha1的值,更新公式的推导如下
        # ---------------------------
        #y1*alpha1_new + y2*alpha2_new_clipped = y1*alpha1_old + y2*alpha2_old
        # ⇒ alpha1_new = alpha1_old + s*(alpha2_old - alpha2_new_clipped)  where s = y1*y2
        #-----------------------------
        a1 = alpha1 + s*(alpha2 - a2)
        # 更新bias
        # ----------------------------
        # Reference:https://blog.csdn.net/luoshixian099/article/details/51227754
        # ----------------------------
        b1 = -E1 - y1*(a1 - alpha1)*K11 - y2*(a2 - alpha2)*K12 + self.b
        b2 = -E2 - y1*(a1 - alpha1)*K12 - y2*(a2 - alpha2)*K22 + self.b
        if  0 < a1 < self.C:
            # 若a1是支持向量,则用a1更新b1
            self.b = b1
        elif 0 < a2 < self.C:
            # 若a2是支持向量,则用a2更新b2
            self.b = b2
        else:
            # 否则用b1与b2的中间值更新
            # ---------------------------------------
            # Reference: https://www.zhihu.com/question/38932229
            # ----------------------------------------
            self.b = b1 + b2
        # 更新缓存的lagrange乘子
        self.alpha[i1] = a1
        self.alpha[i2] = a2
        # 更新non-bound subset所对应的index.
        self.non_bound_subset_index = np.argwhere((0 < self.alpha) & (self.alpha < self.C)).flatten()
        # 用新的乘子对self.Ei进行更新
        # print("update Ei.")
        self.__updateEi()
        # self.Ei[i1] = self.predict_single_sample(self.X[i1]) - self.y[i1]
        # self.Ei[i2] = self.predict_single_sample(self.X[i2]) - self.y[i2]
        return 1
    
    def examineExample(self, i2):
        '''
        Used to check whether the sample corresponding to the incoming index violates KKT conditions.

        -------------------------------------
            Parameters
                i2 : Samples to be tested for violation of KKT conditions.
        '''
        # 通过index获取对应的标签和lagrange乘子
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        # 获取E2 where E2 = f(x2) - y2
        E2 = self.Ei[i2]
        # r2 = y2*E2 = y2*f(x2) - y2² = y2*f(x2) - 1
        r2 = y2*E2
        """
        首先在alpha的更新中会对alpha进行clipped,保证了任意时刻,有 0 ≤ alpha ≤ C
        其次在容忍误差范围内满足KKT条件的表达式与违反KKT条件的表达式如下:
        alpha2 ∈ (0, C) ⇒ -tol < y2*E2 < tol
        alpha2 = 0 ⇒ y2*E2 > -tol
        alpha2 = C ⇒ y2*E2 < tol
        ⇒
            alpha2 ∈ (0, C) ⇒ y2*E2 < -tol | y2*E2 > tol
            alpha2 = 0 ⇒ y2*E2 < -tol
            alpha2 = C ⇒ y2*E2 > tol
        ⇒
            alpha2 > 0 ⇒ y2*E2 > tol   注意这一条件包含了alpha2 == C并违反KKT条件的部分
            alpha2 < C ⇒ y2*E2 < -tol   这一条件包含了alpha2 == 0并违反KKT条件的部分
        """
        # 于是有如下违背KKT条件的判断(外层循环的alpha2必须选择违反KKT条件的支持向量,否则没有更新的必要。)
        if  ((0 < alpha2) & (r2 > self.tol)) or ((alpha2 < self.C) & (r2 < -self.tol)):
            # 进入循环,说明alpha2已经确定,下面确定alpha1(首先在non-bound subset中寻找alpha1)
            # 若non-bound-subset的元素个数大于1,则使用heuristic的方法在non-bound-subset中确定alpha1
            if self.non_bound_subset_index.shape[0] > 1:
                # E2不论正负,选取E1时,只需找到Ei中最大最小元
                # 取i1 = argmax(|maxEi - E2|, |minEi - E2|)
                maxEi, max_index = self.Ei[self.non_bound_subset_index].max(), np.argmax(self.Ei[self.non_bound_subset_index])
                minEi, min_index = self.Ei[self.non_bound_subset_index].min(), np.argmin(self.Ei[self.non_bound_subset_index])
                if abs(maxEi - E2) > abs(minEi - E2):
                    i1 = self.non_bound_subset_index[max_index]
                else:
                    i1 = self.non_bound_subset_index[min_index]
                if self.takeStep(i1, i2):
                    # 进入takeStep部分,尝试更新,若更新成功,则self.takeStep返回1,此时应返回1代表更新成功一次.
                    return 1
                # 若通过|E1 - E2|所找到的alpha1更新失败,则遍历non-bound-subset,依次作为alpha1.
                # loop over all non-zero and non-C alpha
                for i in self.non_bound_subset_index:
                    # 若这里的i与上面由启发式算法选出的最优的i1不相等,再更新,否则直接进行下一次尝试。
                    if i != i1:
                        i1 = i
                        if self.takeStep(i1, i2):
                            # 若更新成功,返回1代表更新成功次数加1.
                            return 1
            # 若在non-bound-subset上全部更新失败或non-bound-subset的支持向量个数少于2个,则遍历整个数据集(除去non-bound-subset)找寻合适的alpha1.
            for i1 in range(self.nrow):
                if i1 not in self.non_bound_subset_index:
                    if self.takeStep(i1, i2):
                        return 1
                
        # 若alpha2不是支持向量或者整个数据集上都没有找到合适的alpha1,返回0代表优化失败.
        return 0
            

    def fit(self, X , y):
        """
        The outer loop keeps alternating between single passes over the entire traingingset and multiple passes over the non-bound subset 
        until the entire training set obey the KKT conditions within eps, whereupon the algorithm terminates.
        --------------------------------------

            Parameters:
                1. X:ndarray
                    The feature matrix of training samples.
                2. y: ndarray
                    The targets of training samples.
        """
        # 获取样本数与特征数
        self.nrow, self.ncol = X.shape

        # 首先将样本标签值转化为-1,1类型的
        # 获取样本的标签可取值
        possible_label = np.unique(y)
        if possible_label.shape[0] != 2:
            raise ValueError("The target of dataset is not binary.")
        # 若样本的标签值不为-1, 1,重置样本的标签
        if set(possible_label) != set([-1, 1]):
            for i in range(self.nrow):
                if y[i] == possible_label[0]:
                    y[i] = -1
                else:
                    y[i] = 1
        # 将数据转化为类属性
        self.X = X
        self.y = y

        # 初始化lagrange乘子
        # 若alpha初始化为0,则更新会很慢。
        self.alpha = np.random.uniform(0, self.C, self.nrow)
        # 初始时self.non_bound_subset = np.array([])
        self.non_bound_subset_index = np.array([])
        # 初始化Ei
        # 最开始时由于alpha都初始化为0，所以直接设置预测标签值(记为label)全部为0
        self.target = self.predict(X)
        self.Ei = self.target - y

        # examineAll控制是否遍历整个数据集
        # numChanged存储一轮循环中优化成功变量的个数
        examineAll = True
        numChanged = 0
        
        iternum = 1
        # outerloop
        while examineAll == True or numChanged > 0:
            print("iternum :", iternum, " numChanged:", numChanged)
            iternum += 1
            # 若numChanged >0,说明上一轮while循环中有优化成功的乘子,
            # 为了该轮循环numChanged能正确反应是否有优化成功的乘子,将numChanged置0.
            if numChanged > 0:
                numChanged = 0
            # 如果examineAll 为真,遍历整个数据集
            if examineAll:
                for i in range(self.nrow):
                    # 检查每一个样本是否可以作为第一个优化变量alpha2
                    numChanged += self.examineExample(i)
            else:
                # 若examineAll != True,则说明进入循环时判定numChanged为正
                # 说明上一轮while循环中,要么遍历了整个数据集完成了乘子更新,要么遍历non-bound subset完成了至少一次乘子的更新.
                # 无论哪一种情形,应在此轮循环中遍历non-bound subset更新乘子.
                for i in self.non_bound_subset_index:
                    # 检查每一个样本是否可以作为第一个优化变量alpha2
                    numChanged += self.examineExample(i)
            if examineAll == True:
                # 若examineAll == True,则刚刚的遍历是在整个数据集上的进行的,此时应设置examineAll == False,
                # 以使下一次遍历在non-bound subset上进行.
                examineAll = False
            elif numChanged == 0:
                # 若numChanged == 0且examineAll = False,说明上一轮的while循环在non-bound subset进行的,并且没有乘子更新成功。
                # 说明non-bound subset上的所有数据都近似满足KKT condition.下一次遍历应在整个数据集上进行。
                examineAll = True
        # 退出outerlooper的唯一一种可能是:examineAll = True 且 numChanged == 0.
        # 此时说明整个数据集上没有严重违反KKT条件的点了,优化完成.

    def __updateEi(self):
        for i in range(self.nrow):
            self.Ei[i] = self.predict_single_sample(self.X[i]) - self.y[i]

    def predict_single_sample(self, x):
        single_target = 0
        for i in np.argwhere(self.alpha != 0).flatten():
            # 计算new_target时,只需要遍历那些alpha != 0的点即可.
            single_target += self.alpha[i]*self.y[i]*self.kernel_(x, self.X[i])
        return single_target + self.b

    def predict(self, X):
        target = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            target[i] = np.sign(self.predict_single_sample(X[i]))
        return target
    
    def score(self, X, y):
        y = y.reshape(-1)
        pred = self.predict(X).reshape(-1)
        return (pred == y).sum()/X.shape[0]

# %%
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
if __name__ == "__main__":
    X, y = make_moons(n_samples = 100,
                                shuffle = True, 
                                noise = .1, 
                                random_state = 42)

    # bulid the model
    model = SVM(kernel = 'rbf', C = 2)
    model.fit(X, y)
    pred = model.predict(X)
    score = model.score(X, y)
    #score = 0.95

# %%
X,y = load_breast_cancer(return_X_y = True)
model = SVM(kernel = 'rbf', C = 2)
model.fit(X, y)
pred = model.predict(X)
score = model.score(X, y)
# score = 1
# %%
