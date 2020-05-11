clear;
 load SMK_CAN_187.mat
whos
[row,col]=size(X);
holdoutCVP = cvpartition(Y,'holdout',0.5);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
xtest=X(test(holdoutCVP),:);
ytest=Y(test(holdoutCVP),:);
N= 19;
feature_freq = zeros(1,19993);
lamba= [0.001 .007 0.6 0.05 0.009];

for k = 1:5
for j = 1:N
holdoutCVP = cvpartition(Y,'holdout',0.5);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);

xtest=X(test(holdoutCVP),:);
ytest=Y(test(holdoutCVP),:);
[y,idx] = datasample(dataTrain',1052,'Replace',false);
newdataTrain=dataTrain(:,idx);
[B,FitInfo] = lasso(newdataTrain,grpTrain','Alpha',lamba(5),'CV',10);
model = B(:,FitInfo.Index1SE)~=0;

for i=1:1052
      if (model(i,1)==1)
      feature_freq (1,idx(i))= feature_freq (1,idx(i)) +1;
      end
end

end
end

probability_matrix= feature_freq/N*k;
threshold = 0.6;

newData=dataTrain(:,(probability_matrix>threshold));


 SVMModel = fitcsvm(newData,grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuracy %%%%%%%%%%
[label,score2] = predict(SVMModel,newData);
correct = 0;
for i=1:93
    if (label(i,1) == grpTrain(i,1))
        correct = correct + 1;
    end
end
total = 93;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%
new_xtest=xtest(:,(probability_matrix>threshold));
[label,score3] = predict(SVMModel,new_xtest);
correct = 0;
for i=1:93
    if (label(i,1) == ytest(i,1))
        correct = correct + 1;
    end
end
total = 93;
Test_accuracy = (correct)*100 / total