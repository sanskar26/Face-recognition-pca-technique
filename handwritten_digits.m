dataset_table = load("Data set/numberlist.mat");
%% storing data tables in vector variables
train_digit= dataset_table.digits_train;
test_digit= dataset_table.digits_test;
train_label= dataset_table.labels_train;
test_label= dataset_table.labels_test;
%% preprocessing of the data
test_digit= double(reshape(test_digit,784,10000));
train_digit= double(reshape(train_digit,784,60000));
m= mean(train_digit')';
processed_data= train_digit-m;

cvr= (processed_data * processed_data')/(60000-1); %calculating covariance
[u,s,v]= svd(cvr); %singular value decomposition,a matlab inbuilt function

eigenV= u(:,1:50); %considering first 50 eigenvectors
projectV1= eigenV'*processed_data;
%% training :
weights = zeros(50,10); %to store weights in matrix
for j= 1:60000
    if train_label(j,1)== 0
        weights(:,10)= weights(:,10)+projectV1(:,j);
    else
        weights(:,train_label(j,1))= weights(:,train_label(j,1))+projectV1(:,j);
    end
end
%% testing of the handwritten numbers:
weights= weights/6000;
projectV2= eigenV'*(test_digit-m);
handwrite_output= 0;
%minimum euclidean distance:
for j= 1:10000
    diff= weights-projectV2(:,j);
    diff= diff.*diff;
    one= ones(1, 50);
    [value,answer]= min(one*diff);
    if answer== test_label(j,1)
        handwrite_output= handwrite_output + 1;
    end
end
%% result
disp(handwrite_output);