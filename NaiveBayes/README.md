naive_bayes class object:

- Multinomial Naive Bayes

      MultionmialNB object:
    
        model principle:
        
            posterior ∝ likeihood*priori
            
            in the following equation.the likeihood P(X(j) = x(j)|y = c_k) and prior P(y = c_k) both obey the multinomial distribution
            
            P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
            
        attribute of instance object:
        
            1.laplacian_smoothing: If you want to use the laplacian_smoothing,you need to keep the parameter as True 
            
                Defalut:Ture
                
            2.lambd: the parameters of laplacian smoothing 
            
                detype:float   Defalut: 1
                
        funtion:
        
            fit():build model by input data.
            
            predict_single_instance():use this function to predict single sample. 
            
                private function of instance object.we can't call this function outside this class
                
            predict(): predict sample by trained model.
            
            score():set accuracy = ΣI(y_predict == ytest) as criterion to measure the goodness of trained model.
            
  -  Gaussian Naive Bayes
  
          GaussianNB object:
        
            model principle:
            
                posterior ∝ likeihood*priori
                
                in the following equation.the likeihood P(X(j) = x(j)|y = c_k) and prior P(y = c_k) both obey the gaussian distribution
                
                P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
                
            funtion:
            
                fit():build model by input data.
                
                __predict_single_instance():use this function to predict single sample. 
                
                    private function of instance object.we can't call this function outside this class
                    
                predict(): predict sample by trained model.
                
                score():set accuracy = ΣI(y_predict == ytest) as criterion to measure the goodness of trained model.
