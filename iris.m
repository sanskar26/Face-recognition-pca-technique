file= fopen("Data set/iris.data");
%tabulating raw data so that be used in matlab
raw_data= split(fscanf(file,"%c"),newline);
tabular_data= split(raw_data(1:150),",");
tabular_data= str2double(tabular_data(:,1:4));% tabular_data now can be used as dataset in matlab
 
concat_tabdata= [tabular_data(1:50,:) tabular_data(51:100,:) tabular_data(101:150,:)];
%% mean and co variance calculation
m1= mean(tabular_data);
m2=mean(concat_tabdata);
Sdeviation1= std(tabular_data);
Sdeviation2= std(concat_tabdata);

final_datatable= (tabular_data-m1)./Sdeviation1;
final_concatdata= (concat_tabdata-m2)./Sdeviation2;

cvr= cov(final_datatable);
concat_cvr= [cov(final_concatdata(:,:,1)) cov(final_concatdata(:,:,1)) cov(final_concatdata(:,:,1))];
[u,s,v] = svd(cvr);
[u1,s1,v1]= svd(concat_cvr(:,:,1));
[u2,s2,v2]= svd(concat_cvr(:,:,1));
[u3,s3,v3]= svd(concat_cvr(:,:,1));

u= u(:,1:2); 
u1= u1(:,1:2); 
u2= u2(:,1:2); 
u3= u3(:,1:2);
%% projection on eigenvectors.
projectV= u'*final_datatable';
projectV1= u1'*final_concatdata(:,:,1)';
projectV2= u2'*final_concatdata(:,:,1)';
projectV3= u3'*final_concatdata(:,:,1)';
%%
%final plot representing all classes of iris.
scatter(projectV(1,1:50),projectV(2,1:50),20,"BLACK")
hold on
scatter(projectV(1,51:100),projectV(2,51:100), 20,"RED")
hold on
scatter(projectV(1,101:150),projectV(2,101:150),20,"BLUE")
hold off
legend('Iris-setosa','Iris-versicolor','Iris-virginica')
%for individual plot of all three classes use:
%scatter(projectionvectorname(1,:),projectvectorname(2,:));
%legend('irisname')