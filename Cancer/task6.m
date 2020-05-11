clear;
load ovariancancer; 
whos
for i = 1: 216
    if (cell2mat(grp(i,1)) == 'Cancer')
        group(i,1) =1;
    else
        group(i,1) = 0;
    end
end
holdoutCVP = cvpartition(group,'holdout',56);
dataTrain = obs(holdoutCVP.training,:);
grpTrain = group(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==0,:);
xtest=obs(test(holdoutCVP),:);
ytest=group(test(holdoutCVP),:);
[B,FitInfo] = lasso(dataTrain,grpTrain,'Alpha',0.75,'CV',10);
model = B(:,FitInfo.Index1SE)~=0;
newData=dataTrain(:,model);

 SVMModel = fitcsvm(newData,grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuracy %%%%%%%%%%
[label,score2] = predict(SVMModel,newData);
correct = 0;
for i=1:160
    if (label(i,1) == grpTrain(i,1))
        correct = correct + 1;
    end
end
total = 160;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%
new_xtest=xtest(:,model);
[label,score3] = predict(SVMModel,new_xtest);
correct = 0;
for i=1:56
    if (label(i,1) == ytest(i,1))
        correct = correct + 1;
    end
end
total = 56;
Test_accuracy = (correct)*100 / total