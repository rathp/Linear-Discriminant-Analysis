function [LDAmodel]= rathp_LDA_train(X_train, Y_train, numofClass)
% Training LDA
%
% EC 503 Learning from Data
% Spring semester, 2016
% by Prakash Ishwar
%
% Assuming D = dimension of data
% Inputs:
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of class 
% Assuming that the classes are labeled  from 1 to numofClass
%
% Output:
% LDAmodel: the parameters of QDA classifier which has the follwoing fields
% Mu : numofClass * D matrix, i-th row = mean vector of class i
% Sigmapooled : D*D  covariance matrix
% pi : numofClass *1 vector, pi(i) = prior probability of class i

% output:
% Y_predict predicted labels for all the testing data points in X_test

% Write your codes here:
mapMatrix=horzcat(X_train,Y_train);
count=zeros(numofClass,1);
[rows,D]=size(X_train);
[rows,r]=size(mapMatrix);
%finding mean vector
labels=unique(Y_train);
numofClass=length(labels);
add=zeros(numofClass,D);
Mu=zeros(numofClass,D);
count=zeros(numofClass,1);

for k=1:numofClass
for i=1:rows
   if mapMatrix(i,D+1)==labels(k)
       count(k)=count(k)+1;
       for j=1:D
           add(k,j)=add(k,j)+X_train(i,j);
           Mu(k,j)=add(k,j)./count(k);
       end
   end
end
end



x_sep=cell(numofClass,1);

 for k=1:numofClass
 temp=zeros(count(k),1);
 temp=x_sep{k,1};
 j=1;
 for i=1:rows
       if mapMatrix(i,D+1)==labels(k);
       temp(j,:)=mapMatrix(i,(1:D));
       j=j+1;
    end
 end
    x_sep{k,1}=temp;
end

Pi=zeros(numofClass,1);


for i=1:numofClass
Pi(i)=count(i)./sum(count);
end


Sigma_t=cell(numofClass,1);


for j=1:numofClass
Sigma_t{j,1}=cov(x_sep{j,1});
end

Sigmapooled=zeros(D);
for i=1:numofClass
Sigmapooled=Pi(i).*(Sigma_t{i})+Sigmapooled;
end



LDAmodel=struct('Mu',Mu,'Sigmapooled',Sigmapooled,'Pi',Pi);

end
