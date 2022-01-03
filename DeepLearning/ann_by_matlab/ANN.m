% training data
filename1 = './train-images.idx3-ubyte';
filename2 = './train-labels.idx1-ubyte';
% testing data
filename3 = './t10k-images.idx3-ubyte';
filename4 = './t10k-labels.idx1-ubyte';
% read data
% train images and labels
Xtrain = read_mnist_images(filename1);
Ytrain = read_mnist_labels(filename2);
% test images aYtrainnd labels
Xtest = read_mnist_images(filename3);
Ytest = read_mnist_labels(filename4);

% Verify that the label corresponds to the image
% imshow(reshape(Xtrain(1,:),28,28)');
% [m,p] = max(Ytrain(1,:));
% p - 1

%min_max_scaler
Xtrain = mapminmax(Xtrain,0,1);
Xtest = mapminmax(Xtest,0,1);

% build the artificial neural network
% The structure of network
%      initialize the parameters of neural network
%      input:x
%      a = activation1(x+b1) 
%      z = Wx+b2
%      output = activation2(z)
%      activation1 : tanh       activation2: softmax
%      loss function: cross entropy
W = sqrt(6/(784+10))*rand(10,28*28) - sqrt(6/(784+10));
b = zeros(1,10);


% the size of Xtrian
[train_size,~] = size(Xtrain);
 % the size of Xtest
[test_size,~] = size(Xtest);
% get the label of test image
[~,p] = max(Ytest,[],2);
% true image label = index - 1
p = p - 1;
% the epoch of train
epoch = 100;
% learning rate
eta = 0.0001;
% accuracy list
accuracy_list = [];
loss = [];
for i = 1:epoch
    for j = 1:train_size
        % activation --- tanh
        a1 = tanh(Xtrain(j,:));
        % function hiden layers
        z = W*a1'+b;
        % activation function ---- softmax
        s = softmax(z);
        % loss ----- cross entropy
        L = -sum(Ytrain(j,:)*log(s));
        % Gradient Desent
        for l = 1:10
            W(l,:) = W(l,:) - eta*(s(l) - Ytrain(j,l))*Xtrain(j,:);
            b(l) = b(l) - eta*(s(l) - Ytrain(j,l));
        end
    end
    
    pre = [];
    %epoch loss in test set
    epoch_loss = 0;
    for k = 1:test_size
        a1 = tanh(Xtest(k,:));
        % function hiden layers
        z = W*a1'+b;
        % activation function ---- softmax
        s = softmax(z);
        % test error
        L = -sum(Ytrain(j,:)*log(s));
        % add to epoch loss
        epoch_loss = epoch_loss + L;
        [~,p1] = max(s);
        pre = [pre;p1];
    end
    pre = pre - 1;
    accuracy = (sum(pre == p)/length(p));
    accuracy_list = [accuracy_list,accuracy];
    loss = [loss,epoch_loss/test_size];
    fprintf('第%d个epoch，test data正确率为%6.2f%%\n',i,accuracy*100);
end

% plot
figure(1)
set(gcf,'position',[150 150 800 800])
suptitle('predict result')
for i = 1:64
    subplot(8,8,i);
    imshow(reshape(Xtest(i,:),28,28)');
    title(num2str(pre(i)))
end

figure(2)
plot(loss);
title('Cross Entropy');
xlabel('Epoch');
ylabel('loss value');
grid on


figure(3)
plot(accuracy_list);
title('Accuracy');
xlabel('Epoch');
ylabel('accuracy');
grid on

