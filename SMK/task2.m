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

mean1= mean(dataTrainG1(:,:));
mean2= mean(dataTrainG2(:,:));
std1=power(std(dataTrainG1(:,:)),2);
std2=power(std(dataTrainG2(:,:)),2);
fisher_score = (power((mean1(:,:)-mean2(:,:)),2));
for i = 1:col
fisher_score(1,i)= fisher_score(1,i)/(std1(1,i)+std(1,i));
end 
[val,ind] = sort(fisher_score,'descend');
topscores=val(1:10);
topscored_features=ind(1:10);

 SVMModel = fitcsvm(dataTrain(:,topscored_features(:,:)),grpTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train data Accuracy %%%%%%%%%%%%%%%%%%%%%
[label,score] = predict(SVMModel,dataTrain(:,topscored_features(:,:)));
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% test data Accuracy %%%%%%%%%%%%%%%%%%%%%
[label,score] = predict(SVMModel,xtest(:,topscored_features(:,:)));
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

        

