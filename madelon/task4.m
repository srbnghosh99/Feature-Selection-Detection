clear;
clc;
load madelon.mat;
whos
[row,col]=size(X);
holdoutCVP = cvpartition(Y,'holdout',100);
dataTrain = X(holdoutCVP.training,:);
grpTrain = Y(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
k = find(grpTrain==-1);
%
dataTrainG2 = dataTrain(k,:);
xtest=X(test(holdoutCVP),:);
ytest=Y(test(holdoutCVP),:);
 

[h,p,ci,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
% showing how well-separated of both grouping by drawing CDF function
ecdf(p);
xlabel('P value');
ylabel('CDF value');
% filter feature selection method and MCE error estimation both on the
% training and testing datasets.
[~,featureIdxSortbyP] = sort(p,2); % sort the features
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
% 10-fold partition for the training set
 tenfoldCVP = cvpartition(grpTrain,'kfold',10);
% use the filter results from the previous section as a pre-processing step to select features. 
fs1 = featureIdxSortbyP(1:150);
% apply forward sequential feature selection on these 150 features
 fsLocal = sequentialfs(classf,dataTrain(:,fs1),grpTrain,'cv',tenfoldCVP);

% The above selected features 
fs1(fsLocal)
% evaluate the performance of the selected model of the above feature on
% the test datasets
testMCELocal = crossval(classf,X(:,fs1(fsLocal)),Y,'partition',holdoutCVP)/holdoutCVP.TestSize
Test_accuracy= (100-testMCELocal*100)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train Accuract
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainMCELocal = crossval(classf,X(:,fs1(fsLocal)),Y,'partition',holdoutCVP)/holdoutCVP.TrainSize
Train_accuracy= (100-trainMCELocal*100)

% the cross-validation MCE as a function of the number of features for up to 50 features
[fsCVfor50,historyCV] = sequentialfs(classf,dataTrain(:,fs1),grpTrain,...
    'cv',tenfoldCVP,'Nf',50);
plot(historyCV.Crit,'o');
xlabel('Number of Features');
ylabel('CV MCE');
title('Forward Sequential Feature Selection with cross-validation');
%The cross-validation MCE reaches the minimum value when 10 features are used 
fsCVfor10 = fs1(historyCV.In(10,:));
% show these 10 features in the order in which they are selected in the sequential forward procedure
[orderlist,ignore] = find( [historyCV.In(1,:); diff(historyCV.In(1:10,:) )]' );
fs1(orderlist)
% evaluate these 10 features, we compute their MCE for QDA on the test set
testMCECVfor10 = crossval(classf,X(:,fsCVfor10),Y,'partition',holdoutCVP)/holdoutCVP.TestSize
% plot of resubstitution MCE values on the training set 
[fsResubfor50,historyResub] = sequentialfs(classf,dataTrain(:,fs1),...
     grpTrain,'cv','resubstitution','Nf',50);
plot(1:50, historyCV.Crit,'bo',1:50, historyResub.Crit,'r^');
xlabel('Number of Features');
ylabel('MCE');
legend({'10-fold CV MCE' 'Resubstitution MCE'},'location','NE');
% Compute the MCE value of these 16 features on the test set to see their real performance
fsResubfor16 = fs1(historyResub.In(16,:));
testMCEResubfor16 = crossval(classf,X(:,fsResubfor16),Y,'partition',holdoutCVP)/holdoutCVP.TestSize

