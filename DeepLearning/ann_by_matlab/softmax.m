%define softmax function
function [y] = softmax(X)
    y = exp(X)/sum(exp(X));
end
 