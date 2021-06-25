# mlimpl
### Machine Learning Implementation
<img src="https://github.com/vincen-github/mlimpl/blob/master/pic/logo.jpeg" width="300" height="300" alt="mlimpl">

![author: vincen (shields.io)](https://img.shields.io/badge/author-vincen-brightgreen) ![email](https://img.shields.io/badge/email-vincen.nwu.%40gmail.com-red) [![zhihu](https://img.shields.io/badge/zhihu-https%3A%2F%2Fwww.zhihu.com%2Fpeople%2Fvincen--43--89-blue)](https://www.zhihu.com/people/vincen-43-89)  ![build: passing (shields.io)](https://img.shields.io/badge/build-passing-brightgreen) ![python: 3.6|3.7|3.8|3.9 (shields.io)](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-blue) ![NumPy version](https://img.shields.io/badge/NumPy-%3E%3D1.19.2-brightgreen) ![Pandas version](https://img.shields.io/badge/Pandas-%3E%3D1.1.3-brightgreen) 

## Introduce
This Repository gathered some Implementation code which encaps-ulates commonly used methods in the field of machine learning based on Numpy and Pandas.u can implement commonly used machine learning algorithms by referring to this repository to deepen your understanding of it.
## trait
- Detailed documentation and comment.
- guidance for error-prone and difficult points.
## Usage
I refer to the class structure of sklearn in my implementation. Most of class has three methods(i.e fit, predict, score).there is an example as follows:
```python
from Multiple_linear_regression import LinearRegression  
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)  
  
reg = LinearRegression()  
reg.fit(X, y)  
y_pred = reg.predict(X)  
score = reg.score(X, y)
```
as u saw. it is same as using sklearn.

## Table of Contents
- Gan:
	- Generate handwritten digital pictures through Gan achieved by tensorflow1.
- Cnn:
	- Recognize digital verification code through convolutional neural network achieved by tensorflow1.
- linear_model:
  - Linear Regression solved by analytical solution/gradient descent/AdamOptimizer.
  - Ridge solved by analytical soulution/gradient descent /AdamOptimizer.
  - Lasso solved by coordinate descent/iterate ridge.
- DecisionTree:
	- ID3: the algorithm to solve classification problem based on tree form.
	- C4.5: Improvement of above method.
                Note that above two methods only support discrete features/labels.
	- Cart: CartRegressor to solve regression problem(i.e continuous label).this code can handle features whether continuous or discrete.
	- On the other hand.there exists some ipynb file implement decision tree which isn't encapsulated as class.
- NaiveBayes:
	- MultinomialNB: Naive Bayes to solve discrete labels that obey multinomial distribution (priori of category).
  - u need to ensure the incoming features are categorical.
                GaussianNB: same as above except priori distribution is Gaussian.it is implies that this method can handle continuous features/label.
- ann_by_matlab:
  - a simple artificial neural network achieved by matlab to distinguish the mnist digital dataset.the code in floder contains artificial neural network implement by myself.The other code file except file named ANN.m is to read mnist dataset to memory throughmatlab.it came from other blog.
 - SVM:
	- Support vector machine solved by sequential minimal optimization algorithm for classification task.
- KMeans++:
	 - Common unsupervised algorithm for cluster improved from kmeans.
- rejection_sampling:
	- Rejection sampling method.
- l1/2(LHalf):
	- l1/2 algorithm is a improved variant algorithm of lasso.it is a linear model as lasso but the optimization object of it is as follows
		                   	<p align="center">min loss = 1/2 ‖Y - Xβ‖ + λ‖β‖_{1/2}</p>
	- i use iterate ridge method to solve this non-convex regularization framework.
	- the file named energy_predict.py is the application of this method in Energy Consumption Field of CNC Machine Tools used pyspark.
- xgboost:
	- eXtreme Gradient Boosting(xgboost) is a class that implement a scalable tree boosting system proposed by TianQi Chen.
	- i implement the exact greedy algorithm/approximate algorithm for split finding in this package.
- RandomForest:
	- A random forest classifier.A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
- GMM:
	- Gaussian Mixture Model(single dimension) solved by EM.
- MCMC：
	- Markov Chain Monte Carlo.It contains Metropolis–Hastings Algorithm and Gibbs Sampling.
	
