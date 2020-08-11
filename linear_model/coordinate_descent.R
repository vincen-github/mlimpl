#lambda为lasso的超参数，eps为迭代终止阈值
coordinate_descent <- function(dataset, lambda = 1, eps = 10^ (-6)) 
{
    #索引行数和列数
    nrow <- dim(dataset)[1]
    ncol <- dim(dataset)[2]
    #索引X与y
    X <- as.matrix(dataset[,-ncol])
    y <- as.matrix(dataset[,ncol])
    #初始化线性模型的系数
    alpha <- rnorm(ncol - 1,mean = 0,sd = 1)
    #alpha的备份，用于存储上一次迭代时alpha的值，已用于比较一次迭代所带来的差异是否大于eps,若非，则跳出循环
    #使用-1的原因是为了保证第一次可以进入下面的while循环
    alpha_copy <- rep(-1,times = ncol - 1)
    # 当alpha与上一轮迭代的alpha的二范数小于eps时，跳出循环，否则继续迭代
    while(sum((alpha - alpha_copy)^2) > eps)
    {
        #将alpha的值赋给alpha_copy,用于记录该轮迭代前的alpha
        alpha_copy <- alpha
        #设l为该轮迭代要更新的alpha的序号
        for(l in 1:(ncol-1))
        {
            #去除alpha的第l个元素
            alpha_drop <- alpha[-l]
            #去除X的第l列
            X_drop <- X[,-l]
            #索引X的第l列
            X_l <- X[,l]
            #计算残差r
            r <- y - (X_drop %*% alpha_drop)
            #计算与软阈值lambda比较的数值
            compare <- 2*(X_l %*% r)/nrow
            #通过compare与软阈值lambda的比较确定alpha_l的更新值
            if(compare > lambda)
            {
                alpha[l] <- nrow*(compare - lambda)/(2*sum(X_l^2))
            }
            else if(compare < -lambda)
            {
                alpha[l] <- nrow*(compare + lambda)/(2*sum(X_l^2))
            }
            else
            {
                alpha[l] = 0
            }
        }
    }
    return(alpha)
}

#用于生成数据的函数,nrow代表生成数据的数量
generate_data = function(beta,nrow = 200)
{
    #所要生成数据矩阵的列数
    ncol <- length(beta)
    #生成随机数矩阵,因需要保留偏置列，故生成的矩阵列数为ncol-1
    X <- matrix(rnorm(nrow*(ncol - 1),mean = 0,sd = 1),nrow = nrow,ncol = ncol - 1)
    #将偏置列加入数据矩阵中
    X <- cbind(X,rep(1,nrow))
    #按beta生成y
    y <- X %*% beta
    dataset <- cbind(X,y)
    return(dataset)
}

#生成数据所使用的beta，最后一项为偏置
beta <- c(1,0,3,0,2,0,2,0,0)

dataset <- generate_data(beta,nrow = 200)
View(dataset)
alpha <- coordinate_descent(dataset,lambda = 0.1,eps = 10^(-6))
print(alpha)
