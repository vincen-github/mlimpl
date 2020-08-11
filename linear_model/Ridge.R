Ridge = function(dataset,C,eta,max_epoch)
{
    nrows <- dim(dataset)[1]
    ncol <- dim(dataset)[2]
    X  <- as.matrix(dataset[,-ncol])
    y <- matrix(dataset[,ncol])
    error_list <- c()
    alpha <- rnorm(ncol - 1, mean=0, sd = 3) 
    epoch = 0
    while(epoch < max_epoch)
    {
        error <- sum((y - X %*% alpha)^2) 
        error_list <- append(error_list,error)
        alpha <- alpha - eta*(-t(X) %*% (y - (X %*% alpha)) + C*alpha)
        epoch <- epoch + 1
    }
    output <- list(one = alpha,two = error_list)
    return(output)
}

df <- read.csv("C:\\Users\\vincen\\Desktop\\boston.csv",header = TRUE,sep = ',')
dim(df)
colnames(df)

result <- Ridge(df,C = 1.6,eta = 0.0001,max_epoch = 2000)
alpha <- result$one
min_error <- min(result$two)
alpha
min_error

nrows <- dim(df)[1]
ncol <- dim(df)[2]
X  <- as.matrix(df[,-ncol])
y <- as.matrix(df[,ncol])
y_predict <- X %*% alpha
plot(y,y_predict,xlab = "y_true",ylab = "y_predict",main = "y_ture VS y_predict") 