clear;
 load SMK_CAN_187.mat
whos
[row,col]=size(X);
holdoutCVP = cvpartition(Y,'holdout',40);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
xtest=X(test(holdoutCVP),:);
ytest=Y(test(holdoutCVP),:);

classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));

[coeff,score] = pca(dataTrain','NumComponents',100);
 tenfoldCVP = cvpartition(grpTrain,'kfold',10);
 fsLocal = sequentialfs(classf,coeff,grpTrain,'cv',tenfoldCVP);
 k = find(fsLocal==1);
 h=X(:,:)*score;
testMCELocal = crossval(classf,h(:,k),Y,'partition',holdoutCVP)/holdoutCVP.TestSize
Test_accuracy= (100-testMCELocal*100)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainMCELocal = crossval(classf,h(:,k),Y,'partition',holdoutCVP)/holdoutCVP.TrainSize
Train_accuracy= (100-trainMCELocal*100)



