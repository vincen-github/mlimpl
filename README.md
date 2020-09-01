# Machine Learning Code

This Repository is python code which packages some  commonly used methods about machine learning method.

I will make subsequent updates to the code.

Specific usage details can be obtained from the source code.

Here is some brief information about code

linear_model:
- LinearRegression
            generate_data(): 按照实例化对象的初始属性生成数据集
            
            regression_cond():计算线性回归中 X 的条件数
            
            regression_correlation():可视化矩阵X的相关系数矩阵
            
            fit():最小二乘估计训练线性模型(直接利用(X^T*X)^(-1)*X*y计算)，return alpha和RSS
            
            score():通过训练好的model计算测试误差
            
            predict():利用已训练好的model预测测试样本
            
            batch_gradient_descent():批量梯度下降 每一个epoch用整个训练集计算梯度
           
            stochastic_gradient_descent():随机梯度下降 每一个epoch中运用每一个样本计算梯度
            
            mini_batch_gradient_descent():小批量梯度下降 每一个epoch中运用等大小的训练集的子集计算梯度
            
            AdamOptimizer():Adaptive Moment Estimation
            
            square_R():goodness of fit
            
            modified_square_R():modified_square_R
            
- Ridge
            fit():最小二乘估计训练线性模型((直接利用(X^T*X)^(-1)+CI)*X*y计算)，return alpha和RSS
            
            batch_gradient_descent():批量梯度下降 每一个epoch用整个训练集计算梯度
            
            stochastic_gradient_descent():随机梯度下降 每一个epoch中运用每一个样本计算梯度
            
            mini_batch_gradient_descent():小批量梯度下降 每一个epoch中运用等大小的训练集的子集计算梯度
            
            AdamOptimizer():Adaptive Moment Estimation
            
            square_R():goodness of fit
            
            modified_square_R():modified_square_R
            
            trace():Ridge coefficients as a function of the regularization
            
 - Lasso
            iterate_ridge:Solve lasso by iterative method of solving ridge multiple times
            
            coordinate_descent():sovle lasso by coordinate descent
      
DecisionTree:
- ID3-C4.5-Cart(Regression),but i did not package it into a class.

NaiveBayes:
- MultinomialNB:Multinomial Naive Bayes
