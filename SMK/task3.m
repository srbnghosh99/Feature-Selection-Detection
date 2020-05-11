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

% dataTrain = dataTrain';
 trainMean = mean(dataTrain');
[coeff,score,mu] = pca(dataTrain','NumComponents',10);

% pca_ten =  transpose(coeff_ten);
% [coeff,score,latent,explained,mu]=pca(dataTrain);

% [coeff,score,latent]=pca(dataTrain);

% trainPCA = pca(dataTrain); 
% numDim = 10;
% [r1,c1]=size(dataTrain)
% [r2,c2]=size(coeff)
% reducedTrainData1= dataTrain *coeff  ;
% reducedTrainData = bsxfun(@minus, dataTrain, trainMean);
% reducedTrainData=mtimes(reducedTrainData,trainPCA(:,1:numDim));
% 
%  [r1,c1]=size( bsxfun(@minus, xtest, trainMean))
%  [r1,c1]= size( coeff(:, 1:10))
%  reducedTestData2 = bsxfun(@minus, xtest, trainMean) * coeff(:, 1:10);
 trainshouldbe=dataTrain*score;
SVMModel = fitcsvm(trainshouldbe,grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Train Accuracy %%%%%%%%%%%%%%%%%%%%%%
[label,score2] = predict(SVMModel,trainshouldbe);
correct = 0;
groupTrain=(grpTrain);
lb=(label);
for i=1:147
    if (lb(i,1) == groupTrain(i,1))
        correct = correct + 1;
    end
end
total = 147;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%%%%%%%%%%%%%%
 testshouldbe= xtest*score;
[label,score3] = predict(SVMModel,testshouldbe);
correct = 0;
yt=(ytest);
lb=(label);
for i=1:40
    if (lb(i,1) == yt(i,1))
        correct = correct + 1;
    end
end
total = 40;
Test_accuracy = (correct)*100 / total

