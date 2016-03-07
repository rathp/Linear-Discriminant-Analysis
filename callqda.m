clc; clear;
Data_set=importdata('C:\Users\home\Desktop\data learning\data_iris.mat');
X_train_in=Data_set.X;
Y_train_in=Data_set.Y;

mapMatrix=horzcat(X_train_in,Y_train_in);

a=randperm(length(mapMatrix),100);%pick 100 samples uniformly from the training set at random
[rows1,r]=size(mapMatrix);
[rows2,D]=size(X_train_in);


map_temp=mapMatrix(a(1:100),(1:r));

b=1:150;
c=ismember(b,a);
j=1;

for i=1:150
if c(i)==0
   d(j)=i;
   j=j+1;
end
end


% f=randperm(length(map_temp),10);
% newmap=map_temp(f(1:10),(1:r));
X_train=map_temp(:,(1:D));
Y_train=map_temp(:,(r));
numofClass=length(unique(Y_train));
[QDAmodel]=QDA_train_pree(X_train,Y_train,numofClass);
[LDAmodel]=LDA_train(X_train,Y_train,numofClass);
% 
% avg10=zeros(3,4);
% avg10=QDAmodel.Mu+avg10
% avg10=(avg10)./10

% avg10(over 10 splits) =
% 
%     5.0130    3.4468    1.4664    0.2468
%     5.9398    2.7485    4.2391    1.3132
%     6.6137    2.9751    5.5448    2.0287

Mean_vector_average=([5.0130 3.4468 1.4664 0.2468;5.9398 2.7485 4.2391 1.3132;6.6137 2.9751 5.5448 2.0287])
% var101=zeros(4);
% var102=zeros(4);
% var103=zeros(4);
% vartemp=cell(numofClass,1);

% vartemp=QDAmodel.Sigma;
% var101=vartemp{1,1}+var101;
% var102=vartemp{1,2}+var102;
% var103=vartemp{1,3}+var103;
% % 
% var101=(var101)./10;
% var102=(var102)./10;
% var103=(var103)./10;
% 
% 
% var101 =
% 
%     0.1298    0.1027    0.0189    0.0123
%     0.1027    0.1463    0.0115    0.0110
%     0.0189    0.0115    0.0330    0.0067
%     0.0123    0.0110    0.0067    0.0114
% 
Variance_QDA_class_1=([0.1298 0.1027 0.0189 0.0123;0.1027 0.1463 0.0115 0.0110;0.0189 0.0115 0.0330 0.0067;0.0123 0.0110 0.0067 0.0114])
% var102 =
% 
%     0.2702    0.0868    0.1883    0.0561
%     0.0868    0.0942    0.0874    0.0429
%     0.1883    0.0874    0.2344    0.0768
%     0.0561    0.0429    0.0768    0.0403
Variance_QDA_class_2=([0.2702 0.0868 0.1883 0.0561;0.0868 0.0942 0.0874 0.0429;0.1883 0.0874 0.2344 0.0768;0.0561 0.0429 0.0768 0.0403])
% var103 =
% 
%     0.3786    0.0845    0.2851    0.0479
%     0.0845    0.0992    0.0709    0.0505
%     0.2851    0.0709    0.2964    0.0529
%     0.0479    0.0505    0.0529    0.0777
Variance_QDA_class_3=([0.3786 0.0845 0.2851 0.0479;0.0845 0.0992 0.0709 0.0505;0.2851 0.0709 0.2964 0.0529;0.0479 0.0505 0.0529 0.0777])
% varl10=zeros(4);
% varlda=LDAmodel.Sigmapooled;
% varl10=varlda+varl10
% varl10=(varl10)./10
% 
% varl10 =
% 
%     0.2480    0.0868    0.1547    0.0352
%     0.0868    0.1057    0.0503    0.0309
%     0.1547    0.0503    0.1686    0.0374
%     0.0352    0.0309    0.0374    0.0378

% check=QDAmodel.Sigma;

map_test1=mapMatrix(d(1:50),(1:r));
[test_rows,test_col]=size(map_test1);

t=randperm(length(map_test1),50);
map_test=map_test1(t(1:50),(1:r));

X_test=map_test(:,(1:D));
Y_test=map_test(:,(D+1:r));

[prediction1]=QDA_test(X_test, QDAmodel, numofClass);
[prediction2]=LDA_test(X_test, LDAmodel, numofClass);


see=horzcat(prediction1,prediction2,map_test(:,r));

CM_QDA=confusionmat(prediction1,map_test(:,r));
CM_LDA=confusionmat(prediction2,map_test(:,r));

% CM_LDA = confusion matrix corresponding to best ccr
% 
%     21     0     0
%      0    11     0
%      0     0    18

% CM_LDA = confusion matrix corresponding to worst ccr
% 
%     14     0     0
%      0    16     2
%      0     1    17


CCR_QDA=trace(CM_QDA)./sum(sum(CM_QDA));
CCR_LDA=trace(CM_LDA)./sum(sum(CM_LDA));

% % sum_ccr_qda=0;
% sum_ccr_qda=CCR_QDA+sum_ccr_qda;
% mean_ccr_qda=sum_ccr_qda./10;
% %0.912
% 
% % sum_ccr_lda=0;
% sum_ccr_lda=CCR_LDA+sum_ccr_lda;
% mean_ccr_lda=sum_ccr_lda./10;
% %0.974
% 
% qda_ccr_vals=([0.94,0.84,0.9,0.96,0.88,0.88,0.92,0.96,0.92,0.92]);
% lda_ccr_vals=([0.98,1,0.98,0.98,0.94,1,0.98,0.96,0.96,0.96]);
% 
% std_qda=std(qda_ccr_vals);%0.0379
% std_lda=std(lda_ccr_vals);%0.190

