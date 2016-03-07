function [Y_predict] = rathp_LDA_test(X_test, LDAmodel, numofClass)
% Testing for LDA
%
% EC 503 Learning from Data
% Spring semester, 2016
% by Prakash Ishwar
%
% Assuming D = dimension of data
% Inputs:
% X_test : test data matrix, each row is a test data point
% numofClass : number of class 
% LDAmodel: the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D*D  covariance 
% LDAmodel.Pi : numofClass *1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
%
% Output:
% Y_predict predicted labels for all the testing data points in X_test

% Write your codes here:


covar1=LDAmodel.Sigmapooled;
muy1=LDAmodel.Mu;
Prob1=LDAmodel.Pi;

% covar1=diag(covt1);
% covar1=diag(covar1);


[test_rows,test_col]=size(X_test);



Dist1=zeros(numofClass,test_rows);
for j=1:test_rows
for i=1:numofClass
%     Dist1(i,j)=((muy1(i,:)*(inv(covar1))))*(X_test(j,:))'-0.5.*(muy1(i,:)*(inv(covar1))*(muy1(i,:))')+log(Prob1(i,1));
      Dist1(i,j)=(muy1(i,:)*(inv(covar1)))*(X_test(j,:))'-0.5.*((muy1(i,:))*(inv(covar1))*(muy1(i,:))')+log((Prob1(i,1)));
%     Dist(i,j)=0.5.*(((X_test(j,:))'-(muy(i,:))')'*(inv(covar{1,i}))*((X_test(j,:))'-(muy(i,:))'))+0.5*log(det(covar{1,i}))-log(Prob(i,1));
end
end

[val,Y_predict]=max(abs(Dist1));

Y_predict=(Y_predict)';



end
