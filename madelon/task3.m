clear;
load madelon.mat;
whos
[row,col]=size(X);
holdoutCVP = cvpartition(Y,'holdout',1000);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==-1,:);
xtest=X(test(holdoutCVP),:);
ytest=Y(test(holdoutCVP),:);
 trainMean = mean(dataTrain');
[coeff,score,mu] = pca(dataTrain','NumComponents',10);

 trainshouldbe=dataTrain*score;
SVMModel = fitcsvm(trainshouldbe,grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Train Accuracy %%%%%%%%%%%%%%%%%%%%%%
[label,score2] = predict(SVMModel,trainshouldbe);
correct = 0;
groupTrain=(grpTrain);
lb=(label);
for i=1:1600
    if (lb(i,1) == groupTrain(i,1))
        correct = correct + 1;
    end
end
total = 1600;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%%%%%%%%%%%%%%
 testshouldbe= xtest*score;
[label,score3] = predict(SVMModel,testshouldbe);
correct = 0;
yt=(ytest);
lb=(label);
for i=1:1000
    if (lb(i,1) == yt(i,1))
        correct = correct + 1;
    end
end
total = 1000;
Test_accuracy = (correct)*100 / total

