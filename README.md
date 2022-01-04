# mlimpl
### Machine Learning Implementation
<img src="./pic/logo.jpg" width="400" height="300" alt="mlimpl">

![author: vincen (shields.io)](https://img.shields.io/badge/author-vincen-brightgreen) ![email](https://img.shields.io/badge/email-vincen.nwu%40gmail.com-red) ![build: passing (shields.io)](https://img.shields.io/badge/build-passing-brightgreen) ![python: >= 3.6 (shields.io)](https://img.shields.io/badge/python-%3E%3D3.6-blue) ![NumPy version](https://img.shields.io/badge/NumPy-%3E%3D1.19.2-brightgreen) ![Pandas version](https://img.shields.io/badge/Pandas-%3E%3D1.1.3-brightgreen)

## Introduce
This Repository gathered some code which encapsulates commonly used methods in the field of machine learning. Most of them are based on Numpy and Pandas.U can deepen u understanding of related algorithm by referring to this repository.
## trait
- Detailed documentation and annotation.
- guidance for difficulty of algorithm.
## Usage
My implementations refer to the class structure in sklearn for reducing learning cost. Most of class has three methods,i.e., fit, predict, score.There is an example as follows:
```python
from Multiple_linear_regression import LinearRegression
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
score = reg.score(X, y)
```
it is same as using sklearn as u saw.

## Table of Contents
***1. Deep Learning***
	This folder contains the code related to deep learning experiment.Most of them are implemented by Pytorch or Tensorflow.
-   **1. Gan**
    - Generative Adversarial Networks(Gan) implementation using tensorflow1 and apply it to generate mnist dataset.
-   **2. Cnn**
  	- Convolutional Neural Network implemented by  tensorflow1 to recognize digital verification code.
- **3. ann_by_matlab**
    - A toy artificial neural network implementation using matlab to classify mnist dataset.
    - The contents in this floder are implemented by myself expect the program which is for reading mnist dataset to memory.
- **High Confidence Predictions for Unrecognizable Images**
	- A simple demo about adversarial examples.
	- Reference: *Anh Nguyen, Jason Yosinski and Jeff Clune. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 427-436.*
  
***2. Traditional Machine Learning***
-   **1. linear_model**
    - Linear Regression
        - Analytical solution.
  			<center><img src="./pic/linear_reg_formula.png" width="180" height="30"><center>
        - Gradient Descent.
  			<center><img src="./pic/gradient_dec_formula.png" width="180" height="30"><center>
        - AdamOptimizer.
          - Reference: *Sebastian Ruder. An overview of gradient descent optimization algorithms. CoRR, abs/1609.04747,2016.*
    - Ridge
      - Samilar as above.
    - Lasso
      - Coordinate Descent.
      - Iterated Ridge Regression.
        - using following approximation of $\ell_1$ norm to tranform lasso to iterated Ridge quesition.
				<center><img src="./pic/iterated_ridge_formula.png" width="100" height="50"><center>
		- Reference: *Mark Schmidt. Least squares optimization with l1-norm regularization. CS542B Project Report,504:195–221, 2005.*
-   **2. DecisionTree**
    - ID3
      - Using information gain as criterion of buliding tree,whose formulaes are
				<center><img src="./pic/id3_formula.png" width="230" height="150"><center>
    - C4.5
  		- Improvement of above method.Change the criterion to information gain ratio which has the following form
  				<center><img src="./pic/c4.5_formula.png" width="150" height="45"><center>
      - ***Note that above two implementation only support discrete features/labels.***
	- Cart
    	-  CartRegressor is to solve regression problem.This implementation can handle both continuous and discrete feature.Intrinsical optimal question is
				<center><img src="./pic/cart_formula.png" width="400" height="70"><center>
		- For more details, please refer to *统计学习方法—李航*.
    - ***There exists some decision tree implementation in ipynb file,which isn't encapsulated as class.***
- **3. NaiveBayes**
	- MultinomialNB
    	-  Naive Bayes to solve discrete labels whose priori obey multinomial distribution.
         - U need to ensure the incoming features are discrete that ensure this implementation effective.
       -  For more details refer to *统计学习方法—李航*.
    - GaussianNB
        -  Simalar mehtod as above but priori and likelihood are obey Gaussian.This implies that this method can handle continuous features/label.
        -  For more details refer to *机器学习—周志华*.
 - **4. SVM**
	- Support vector machine solved by sequential minimal optimization algorithm for classification task.
	- Reference:
    	- [1] *John Platt. Sequential minimal optimization: A fast algorithm for training support vector machines. Technical Report MSR-TR-98-14, Microsoft, April 1998.*
    	- [2] *统计学习方法—李航*.

- **5. KMeans++**
	 - Common unsupervised algorithm for cluster improved from kmeans.
    - *For more details refer to KMeans documentation in sklearn.*
- **6. rejection_sampling**
	- Rejection sampling method.A effective method to sample from a complex distribution.
- **7. l1/2**
	- l1/2 algorithm is a improved variant algorithm of lasso. it is a linear model as lasso but the optimization object of it is
			<center><img src="./pic/lhalf_formula_1.png" width="280" height="50"><center>
    	- Similar with the method for solveing lasso, i.e., iterated ridge regression.The way for solving this non-convex regularization framework is to transform it to iterated lasso or ridge regression as below. 
    		<center><img src="./pic/lhalf_formula_2.png" width="110" height="55"><center>

		- Reference: *Xu, Z., Chang, X., Xu, F., & Zhang, H. (2012). L1/2 regularization: a thresholding representation theory and a fast solver. IEEE transactions on neural networks and learning systems, 23(7), 1013–1027. https://doi.org/10.1109/TNNLS.2012.2197412*
	- the file named energy_predict.py is the distributed application on Energy Consumption Field of CNC Machine Tools.Distributed platform using here is Spark.
- **8. xgboost**
	- eXtreme Gradient Boosting(xgboost) is a class that implement a scalable tree boosting system proposed by TianQi Chen.
        - i have implemented the exact greedy and approximate algorithm for split finding.
    	- Reference:*Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. CoRR,abs/1603.02754, 2016.*
- **9. RandomForest**
	- A random forest classifier.A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
- **10. GMM**
	- Gaussian Mixture Model(single dimension) solved by EM.
	- Reference:[Expectation Maximization-Yi Da Xu](https://github.com/roboticcam/machine-learning-notes/blob/master/files/em.pdf)
- **11. MCMC**
	- Markov Chain Monte Carlo.
    	-  Metropolis–Hastings Algorithm
    	-  Gibbs Sampling(To Be Completed).
	- Reference:[Markov Chain Monte Carlo-Yi Da Xu](https://github.com/roboticcam/machine-learning-notes/blob/master/files/markov_chain_monte_carlo.pdf.)

