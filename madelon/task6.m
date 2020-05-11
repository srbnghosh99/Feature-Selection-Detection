clear;
load madelon.mat;
whos
[row,col]=size(X);
holdoutCVP = cvpartition(Y,'holdout',1000);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
k = find(grpTrain==-1);
% grpTrain(k,1)=0;
dataTrainG2 = dataTrain(k,:);
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
for i=1:1600
    if (label(i,1) == grpTrain(i,1))
        correct = correct + 1;
    end
end
total = 1600;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%
new_xtest=xtest(:,model);
[label,score3] = predict(SVMModel,new_xtest);
correct = 0;
% yt=cell2mat(ytest);
% lb=cell2mat(label);
for i=1:1000
    if (label(i,1) == ytest(i,1))
        correct = correct + 1;
    end
end
total = 1000;
Test_accuracy = (correct)*100 / total