%% Selecting Features for Classifying High-dimensional Data
% This example shows how to select features for classifying
% high-dimensional data. More specifically, it shows how to perform
% sequential feature selection, which is one of the most popular feature
% selection algorithms. It also shows how to use holdout and
% cross-validation to evaluate the performance of the selected features.
%
% Reducing the number of features (dimensionality) is important in 
% statistical learning. For many data sets with a large number of features
% and a limited number of observations, such as bioinformatics data,
% usually many features are not useful for producing a desired learning
% result and the limited observations may lead the learning algorithm to
% overfit to the noise. Reducing features can also save storage and
% computation time and increase comprehensibility.
%
% There are two main approaches to reducing features: feature selection and
% feature transformation.  Feature selection algorithms select a subset of
% features from the original feature set; feature transformation
% methods transform data from the original high-dimensional feature space
% to a new space with reduced dimensionality. 

%   Copyright 2007-2014 The MathWorks, Inc.

%% Loading the Data
% Serum proteomic pattern diagnostics can be used to differentiate
% observations from patients with and without disease. Profile patterns are
% generated using surface-enhanced laser desorption and ionization (SELDI)
% protein mass spectrometry. These features are ion intensity levels at
% specific mass/charge values.
%
% The data in this example is from the
% <https://home.ccr.cancer.gov/ncifdaproteomics/ppatterns.asp FDA-NCI
% Clinical Proteomics Program Databank>. This example uses the
% high-resolution ovarian cancer data set that was generated using the WCX2
% protein array. After some pre-processing steps, similar to those shown in
% the Bioinformatics Toolbox(TM) example
% <https://www.mathworks.com/help/bioinfo/examples/preprocessing-raw-mass-spectrometry-data.html 
% Pre-processing Raw Mass Spectrometry
% Data>, the data set has two variables |obs| and |grp|. The |obs|
% variable consists 216 observations with 4000 features. Each element in
% |grp| defines the group to which the corresponding row of |obs| belongs.

load ovariancancer; 
whos
holdoutCVP = cvpartition(grp,'holdout',56)
%%
dataTrain = obs(holdoutCVP.training,:);
grpTrain = grp(holdoutCVP.training);

dataTest = obs(holdoutCVP.testing,:);
grptest = grp(holdoutCVP.testing);
try
   yhat = classify(obs(test(holdoutCVP),:), dataTrain, grpTrain,'quadratic');
catch ME
   display(ME.message);
end

dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
[h,p,ci,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
%%
% In order to get a general idea of how well-separated the two groups are
% by each feature, we plot the empirical cumulative distribution function
% (CDF) of the _p_-values:
ecdf(p);
xlabel('P value'); 
ylabel('CDF value') 
 

[~,featureIdxSortbyP] = sort(p,2); % sort the features
testMCE = zeros(1,14);
resubMCE = zeros(1,14);
nfs = 5:5:70;
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
resubCVP = cvpartition(length(grp),'resubstitution')         
for i = 1:14
   fs = featureIdxSortbyP(1:nfs(i));
   testMCE(i) = crossval(classf,obs(:,fs),grp,'partition',holdoutCVP)...
       /holdoutCVP.TestSize;
   resubMCE(i) = crossval(classf,obs(:,fs),grp,'partition',resubCVP)/...
       resubCVP.TestSize;
end
 plot(nfs, testMCE,'o',nfs,resubMCE,'r^');
 xlabel('Number of Features');
 ylabel('MCE');
 legend({'MCE on the test set' 'Resubstitution MCE'},'location','NW');
 title('Simple Filter Feature Selection Method');

%% 
% For convenience, |classf| is defined as an anonymous function. It fits
% QDA on the given training set and returns the number of misclassified
% samples for the given test set. If you were developing your own
% classification algorithm, you might want to put it in a separate file, as
% follows:

%  function err = classf(xtrain,ytrain,xtest,ytest)
%       yfit = classify(xtest,xtrain,ytrain,'quadratic');
%        err = sum(~strcmp(ytest,yfit));

%%
% The resubstitution MCE is over-optimistic. It consistently decreases when
% more features are used and drops to zero when more than 60 features are
% used. However, if the test error increases while the resubstitution error
% still decreases, then overfitting may have occurred. This simple filter
% feature selection method gets the smallest MCE on the test set when 15
% features are used. The plot shows overfitting begins to occur when 20 or
% more features are used. The smallest MCE on the test set is 12.5%:
testMCE(3)
%%
% These are the first 15 features that achieve the minimum MCE:
featureIdxSortbyP(1:15)

%% Applying Sequential Feature Selection

corr(dataTrain(:,featureIdxSortbyP(1)),dataTrain(:,featureIdxSortbyP(2)))
%%
% This kind of simple feature selection procedure is usually used as a
% pre-processing step since it is fast. More advanced feature selection
% algorithms improve the performance. Sequential feature selection is one
% of the most widely used techniques. It selects a subset of features by
% sequentially adding (forward search) or removing (backward search) until
% certain stopping conditions are satisfied. 
%
% In this example, we use forward sequential feature selection in a wrapper
% fashion to find important features. More specifically, since the typical
% goal of classification is to minimize the MCE, the feature selection
% procedure performs a sequential search using the MCE of the learning
% algorithm QDA on each candidate feature subset as the performance
% indicator for that subset. The training set is used to select the
% features and to fit the QDA model, and the test set is used to evaluate
% the performance of the finally selected feature. During the feature
% selection procedure, to evaluate and to compare the performance of the
% each candidate feature subset, we apply stratified 10-fold
% cross-validation to the training set. We will illustrate later why
% applying cross-validation to the training set is important.
%
% First we generate a stratified 10-fold partition for the training set:
tenfoldCVP = cvpartition(grpTrain,'kfold',10) 
%%
% Then we use the filter results from the previous section as a
% pre-processing step to select features. For instance, we select 150
% features here:
fs1 = featureIdxSortbyP(1:150);
%%
% We apply forward sequential feature selection on these 150 features.
% The function |sequentialfs| provides a simple way (the default option) to
% decide how many features are needed. It stops when the first local
% minimum of the cross-validation MCE is found.
 fsLocal = sequentialfs(classf,dataTrain(:,fs1),grpTrain,'cv',tenfoldCVP);
%%
% The selected features are the following:
fs1(fsLocal)

%%
% To evaluate the performance of the selected model with these three features,
% we compute the MCE on the 56 test samples.
testMCELocal = crossval(classf,obs(:,fs1(fsLocal)),grp,'partition',...
    holdoutCVP)/holdoutCVP.TestSize
%% 
% With only three features being selected, the MCE is only a little over
% half of the smallest MCE using the simple filter feature selection method.

%%
% The algorithm may have stopped prematurely. Sometimes a smaller MCE is
% achievable by looking for the minimum of the cross-validation MCE over a
% reasonable range of number of features. For instance, we draw the plot of
% the cross-validation MCE as a function of the number of features for up
% to 50 features.
[fsCVfor50,historyCV] = sequentialfs(classf,dataTrain(:,fs1),grpTrain,...
    'cv',tenfoldCVP,'Nf',50);
plot(historyCV.Crit,'o');
xlabel('Number of Features');
ylabel('CV MCE');
title('Forward Sequential Feature Selection with cross-validation');
%%
% The cross-validation MCE reaches the minimum value when 10 features are used
% and this curve stays flat over the range from 10 features to 35 features.
% Also, the curve goes up when more than 35 features are used, which means
% overfitting occurs there.
%
% It is usually preferable to have fewer features, so here we pick 10
% features:
fsCVfor10 = fs1(historyCV.In(10,:))
%% 
% To show these 10 features in the order in which they are selected in the
% sequential forward procedure, we find the row in which they first become
% true in the |historyCV| output:
[orderlist,ignore] = find( [historyCV.In(1,:); diff(historyCV.In(1:10,:) )]' );
fs1(orderlist)
%%
% To evaluate these 10 features, we compute their MCE for QDA on the test
% set. We get the smallest MCE value so far:
testMCECVfor10 = crossval(classf,obs(:,fsCVfor10),grp,'partition',...
    holdoutCVP)/holdoutCVP.TestSize 

%%
% It is interesting to look at the plot of resubstitution MCE values on the
% training set (i.e., without performing cross-validation during the
% feature selection procedure) as a function of the number of features:
[fsResubfor50,historyResub] = sequentialfs(classf,dataTrain(:,fs1),...
     grpTrain,'cv','resubstitution','Nf',50);
plot(1:50, historyCV.Crit,'bo',1:50, historyResub.Crit,'r^');
xlabel('Number of Features');
ylabel('MCE');
legend({'10-fold CV MCE' 'Resubstitution MCE'},'location','NE');
%%
% Again, the resubstitution MCE values are overly optimistic here. Most are
% smaller than the cross-validation MCE values, and the resubstitution MCE
% goes to zero when 16 features are used. We can compute the MCE value of
% these 16 features on the test set to see their real performance:
fsResubfor16 = fs1(historyResub.In(16,:));
testMCEResubfor16 = crossval(classf,obs(:,fsResubfor16),grp,'partition',...
    holdoutCVP)/holdoutCVP.TestSize 
%%
% |testMCEResubfor16|, the performance of these 16 features (chosen by
% resubstitution during the feature selection procedure) on the test set, is
% about double that for |testMCECVfor10|, the performance of the 10 features
% (chosen by 10-fold cross-validation during the feature selection procedure)
% on the test set. It again indicates that the resubstitution error generally
% is not a good performance estimate for evaluating and selecting features. We
% may want to avoid using resubstitution error, not only during the final
% evaluation step, but also during the feature selection procedure.
