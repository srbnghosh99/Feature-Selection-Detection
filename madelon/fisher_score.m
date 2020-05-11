dataTrain = obs(holdoutCVP.training,:);
grpTrain = grp(holdoutCVP.training);
dataTrainG1 = dataTrain(grp2idx(grpTrain)==1,:);
dataTrainG2 = dataTrain(grp2idx(grpTrain)==2,:);
xtest=obs(test(holdoutCVP),:);
ytest=grp(test(holdoutCVP),:);

mean1= mean(dataTrainG1(:,:));
mean2= mean(dataTrainG2(:,:));
std1=power(std(dataTrainG1(:,:)),2);
std2=power(std(dataTrainG2(:,:)),2);
fisher_score = (power((mean1(:,:)-mean2(:,:)),2));
for i = 1:4000
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
groupTrain=cell2mat(grpTrain);
lb=cell2mat(label);
for i=1:160
    if (lb(i,1) == groupTrain(i,1))
        correct = correct + 1;
    end
end
total = 160;
accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% test data Accuracy %%%%%%%%%%%%%%%%%%%%%
[label,score] = predict(SVMModel,xtest(:,topscored_features(:,:)));
testMCE = zeros(1,10);
resubMCE = zeros(1,10);
correct = 0;
yt=cell2mat(ytest);
lb=cell2mat(label);
for i=1:56
    if (lb(i,1) == yt(i,1))
        correct = correct + 1;
    end
end
total = 56;
accuracy = (correct)*100 / total

        

