clear;
load ovariancancer; 
whos
holdoutCVP = cvpartition(grp,'holdout',56);
dataTrain = obs(holdoutCVP.training,:);
grpTrain = grp(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
xtest=obs(test(holdoutCVP),:);
ytest=grp(test(holdoutCVP),:);
% dataTrain = dataTrain';
% [h,p,ci,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
% % showing how well-separated of both grouping by drawing CDF function
% ecdf(p);
% xlabel('P value');
% ylabel('CDF value');
% filter feature selection method and MCE error estimation both on the
% training and testing datasets.
% [~,featureIdxSortbyP] = sort(p,2); % sort the features
%  trainMean = mean(dataTrain');

classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));

[coeff,score] = pca(dataTrain','NumComponents',100);
 tenfoldCVP = cvpartition(grpTrain,'kfold',10);
 fsLocal = sequentialfs(classf,coeff,grpTrain,'cv',tenfoldCVP);
 k = find(fsLocal==1);
 h=obs(:,:)*score;
testMCELocal = crossval(classf,h(:,k),grp,'partition',holdoutCVP)/holdoutCVP.TestSize

Test_accuracy= (100-testMCELocal*100)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainMCELocal = crossval(classf,h(:,k),grp,'partition',holdoutCVP)/holdoutCVP.TrainSize
Train_accuracy= (100-trainMCELocal*100)



