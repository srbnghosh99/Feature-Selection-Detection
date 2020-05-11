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

[B,FitInfo] = lasso(dataTrain,grpTrain,'Alpha',0.75,'CV',10);
model = B(:,FitInfo.Index1SE)~=0;
newData=dataTrain(:,model);

 SVMModel = fitcsvm(newData,grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuracy %%%%%%%%%%
[label,score2] = predict(SVMModel,newData);
correct = 0;
for i=1:147
    if (label(i,1) == grpTrain(i,1))
        correct = correct + 1;
    end
end
total = 147;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%
new_xtest=xtest(:,model);
[label,score3] = predict(SVMModel,new_xtest);
correct = 0;
for i=1:40
    if (label(i,1) == ytest(i,1))
        correct = correct + 1;
    end
end
total = 40;
Test_accuracy = (correct)*100 / total