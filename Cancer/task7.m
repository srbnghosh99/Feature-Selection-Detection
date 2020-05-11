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
holdoutCVP = cvpartition(group,'holdout',0.5);
dataTrain = obs(holdoutCVP.training,:);
grpTrain = group(holdoutCVP.training);
xtest=obs(test(holdoutCVP),:);
ytest=group(test(holdoutCVP),:);
% 
% [r1,c1]=size(dataTrain')
% [r2,c2]=size(grpTrain)
N= 100;
feature_freq = zeros(1,4000);
lamba= [0.001 .007 0.6 0.05 0.009];

for k = 1:5
count = 0;
for j = 1:N
holdoutCVP = cvpartition(group,'holdout',0.5);
dataTrain = obs(holdoutCVP.training,:);
grpTrain = group(holdoutCVP.training);

xtest=obs(test(holdoutCVP),:);
ytest=group(test(holdoutCVP),:);
[y,idx] = datasample(dataTrain',40,'Replace',false);
newdataTrain=dataTrain(:,idx);

[B,FitInfo] = lasso(newdataTrain,grpTrain','Alpha',lamba(k),'CV',10);


model = B(:,FitInfo.Index1SE)~=0;
% jog=sum(model)
for i=1:40
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
for i=1:108
    if (label(i,1) == grpTrain(i,1))
        correct = correct + 1;
    end
end
total = 108;
Train_accuracy = (correct)*100 / total

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Accuracy %%%%%%%%%%
new_xtest=xtest(:,(probability_matrix>threshold));
[label,score3] = predict(SVMModel,new_xtest);
correct = 0;
for i=1:108
    if (label(i,1) == ytest(i,1))
        correct = correct + 1;
    end
end
total = 108;
Test_accuracy = (correct)*100 / total