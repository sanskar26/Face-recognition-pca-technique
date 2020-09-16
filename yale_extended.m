%loading example data:
load("Data set/face data/YaleB_32x32.mat");
face_feature= fea'; %storing face images as a 1024*2414 matrix
labels= gnd'; %storing labels as a 1*2414 vector

random_list= randperm(size(labels,2),100); %randomly choosing 100 indices to be the testing dataset  
random_list= sort(random_list,'descend');
%%
%initialising data:
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

images_num= size(labels,2); %training images= 2414
images_num = images_num - 100;
m = mean(train_fea')';
train_fea = train_fea - m;

cvr= (train_fea * train_fea')/(images_num - 1); % calculating the covariance matrix
[a,b,v]= svd(cvr); % singular value decomposition where a has eigenvectors and b is the diagonal matrix of eigenvalues.

k= 60;
eigenV= a(:,4:k+3); %taking first k eigenvectors(It is customisable)

%feature extraction:
projectV1= eigenV'*train_fea; % Projecting training data on eigenvectors

weights= zeros(k,38); %weight matrix
total_images = zeros(38); % total images in training set matrix

for j= 1:images_num
    index= train_labels(j);
    total_images(index)= total_images(index)+1;
    weights(:,index)= weights(:,index)+projectV1(:,j);
end

for j = 1:38
    weights(:,j)= weights(:,j)/total_images(j);
end

%Testing of the remaining images
projectV2= eigenV'*(test_data - m);
true_accuracy = 0;
%minimum euclidean distance:
for j= 1:100
    diff= weights-projectV2(:,j);  
    diff= diff.*diff;              
    one= ones(1,60);
    [value,answer]= min(one*diff);  
    if answer== test_labels(j)
        true_accuracy= true_accuracy + 1;
    end
end
fprintf('The accuracy achieved is %d %% \n',true_accuracy);