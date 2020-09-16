addpath('simpleNN')
addpath('simpleNN/library')

%loading example data:
load("Data set/face data/YaleB_32x32.mat"); 
face_feature= fea'; %storing face images as a 1024*2414 matrix
labels= gnd'; %storing labels as a 1*2414 vector

images_num= size(labels,2); %training images= 2414       
random_list= randperm(images_num,100); %randomly choosing 100 indices to be the testing dataset
random_list= sort(random_list,'descend');

train_fea= face_feature; 
test_data= [];
train_labels= labels; 
test_labels= [];

for j= 1:100
    train_fea(:,random_list(j))= [];
    train_labels(:,random_list(j))= [];
    test_data= [face_feature(:,j) test_data];
    test_labels= [labels(:,j) test_labels];
end

images_num= images_num-100;
m= mean(train_fea')';
train_fea= train_fea - m;

cvr= (train_fea*train_fea')/(images_num-1); % calculating the covariance matrix
[a,b,v] = svd(cvr); % singular value decomposition where a has eigenvectors and b is the diagonal matrix of eigenvalues.

k= 60;
eigenV= a(:,4:k+3); %taking first k eigenvectors(It is customisable)

%feature extraction:
projectV1 = eigenV'*train_fea; % Projecting training data on eigenvectors
%% Settings
%using slight customised options:
nnOptions= {'lambda',0.1,'activationFn','tanh','validPercent',30};
%% Run the training
modelNN = learnNN(projectV1',train_labels',nnOptions);%modelNN:The trained NN model
% plotting the confusion matrix for validation set
figure(1); cla(gca);
plotConfMat(modelNN.confusion_valid); %plotConfMat is a matlab function that plots the confusion matrix for the validation set.
%%
% Testing of the remaining images:
projectV2= eigenV'*(test_data-m);% Projecting test data on eigenvectors
true_accuracy= 0;
%%
%prediction:
for j= 1:100
    k= predictNN(projectV2(:,j)',modelNN);
    if k== test_labels(j)
        true_accuracy= true_accuracy+1;
    end
end
fprintf('The accuracy achieved is %d %% \n',true_accuracy);