# Machine Learning Code

This Repository is some code which packages commonly used methods in machine learning.

I will make subsequent updates to the code.

u can obtain the specific usage details from the source code.

Here is some brief infoimation about the main work in each folder.

- Gan:
        Generate handwritten digital pictures through Gan achieved by tensorflow1.

- Cnn:
    Recognize digital verification code through convolutional neural network achieved by tensorflow1.
    u can use it to solve the obstacle of the verification code to the automated crawler.
    note that i apply the python code from web for generate verification code as training/testing dataset.
  
- linear_model:
    Linear Regression solved by analytical solution/gradient descent.
    Ridge solved by analytical soulution/gradient descent/AdamOptimizer.
    Lasso solved by coordinate descent/iterate ridge.

- DecisionTree:
    ID3: the algorithm to solve classification problem based on tree form.
    C4.5: Improvement of above method.
    Note that above two methods only support discrete features/labels.
    Cart: CartRegressor to solve regression problem(i.e continuous label).this code can handle features whether continuous or discrete.
  
- NaiveBayes:
    MultinomialNB: Naive Bayes to solve discrete labels that obey multinomial distribution(priori of category).u need to ensure the incoming features are categorical.
    GaussianNB: same as above except priori distribution is Gaussian.it is implies that this method can handle continuous features/label.

- ann_by_matlab:  
    a simple artificial neural network achieved by matlab to distinguish the mnist digital dataset.
    the code in floder contains artificial neural network implement by myself.
    The other code file except file named ANN.m is to read mnist dataset to memory through matlab.This is not what I did by myself.

- SVM: 
    Support vector machine solved by sequential minimal optimization algorithm for classification task.

- KMeans++: 
    Common unsupervised algorithm for cluster improved from kmeans.
  
- l1/2(LHalf):
    1/2 algorithm is a improved variant algorithm of lasso.
        it is a linear model as lasso but the optimization object of it is as follows
                    min loss = 1/2*||Y - Xβ|| + λ||β||_{1/2}
        Where the form of regular term ||β||_{1/2} as follows
                    ||β||_{1/2} = Σβ_i^{1/2}
    i use iterate ridge method to solve this non-convex regularization framework.
    the file named energy_predict.py is the application of this method in Energy Consumption Field of CNC Machine Tools used pyspark.

- xgboost:
    eXtreme Gradient Boosting(xgboost) is a class that implement a scalable tree boosting system proposed by TianQi Chen.
    i implement the exact exact greedy algorithm/approximate algorithm for split finding in this package.
    
 
