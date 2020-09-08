# Machine Learning Code

This Repository is some code which packages commonly used methods about machine learning method.

I will make subsequent updates to the code.

Specific usage details can be obtained from the source code.

Here is some brief information about code

linear_model:
- LinearRegression

            generate_data(): Generate data according to the initial properties of the instantiated object
            
            fit():training linear model (calculate by (X^T*X)^(-1)X*y), return alpha and RSS
            
            score():Calculate the test error through the trained model, return SSE
            
            predict():Use the trained model to predict the test sample
            
            batch_gradient_descent(): Each epoch uses the entire training set to calculate the gradient
            
            stochastic_gradient_descent():Stochastic gradient descent uses each sample to calculate the gradient in each epoch.
            
            mini_batch_gradient_descent():In each epoch, a subset of the training set of equal size is used to calculate the gradient
            
            AdamOptimizer():Adaptive Movement Estimation
            
            square_R():goodness of fit
            
            modified_square_R():modified_square_R
            
- Ridge

            fit():training linear model (calculate by ((X^T*X)^(-1)+CI)X*y), return alpha and RSS
            
            batch_gradient_descent(): Each epoch uses the entire training set to calculate the gradient
            
            stochastic_gradient_descent():Stochastic gradient descent uses each sample to calculate the gradient in each epoch.
            
            mini_batch_gradient_descent():In each epoch, a subset of the training set of equal size is used to calculate the gradient
            
            AdamOptimizer():Adaptive Moment Estimation
            
            square_R():goodness of fit
            
            modified_square_R():modified_square_R
            
            trace():Ridge coefficients as a function of the regularization
            
 - Lasso
 
            iterate_ridge:Solve lasso by iterative method of solving ridge multiple times
            
            coordinate_descent():sovle lasso by coordinate descent.
      
- DecisionTree:

            ID3.
            
            C4.5.
            
            Cart(Regression).

- NaiveBayes:

            MultinomialNB:Multinomial Naive Bayes.
            
            GaussianNB:Gaussian Naive Bayes.
- Gan:

            Generate handwritten digital pictures through Gan and achieve by tensorflow.

- ann_by_matlab:
            
            a simple artificial neural networks achieved by matlab to distinguish the mnist digital dataset.

- CNN:

            Recognize digital verification code through CNN
