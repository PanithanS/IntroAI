clc
clear all

%% Importing Dataset

%The dataset consists of 150 records of Iris plant with four features: 
% 'sepal-length', 'sepal-width', 'petal-length', and 'petal-width'. 
% All of the features are numeric. 
% The records have been classified into one of the three classes i.e. 'setosa', 'versicolor', or 'verginica'.

dataset_table = readtable('E:\Python_Projects_Git\AI_class\Iris.csv');
features = removevars(dataset_table, {'Id', 'Species'}); % Remove non-numeric columns
features = table2array(features);

labels = removevars(dataset_table, {'Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' });
labels = table2array(labels);

%% Normalization of Features

features_norm = normalize(features, 2, 'norm');

%% Applying PCA

[sa,va,da]=svd(features_norm,'econ');
sa=sa*va;
da=da';

%% PCA Projection

%eigenvalues figure
figure(1)
eigenvalues  = diag(va);
plot(eigenvalues, '-o','markersize',15)
xlabel('Principal componsnets (#)')
ylabel('Eigenvalue (a.u.)')
set(gca,'fontsize',15)

sa = reshape(sa, [50 3 4]);
figure(2)
for i =1:3
plot(sa(:,i,1), sa(:,i,2)*-1, 'o','markersize',8), hold on
end

hold off
ylabel('Principal Component 2','fontsize',15)
xlabel('Principal Component 1','fontsize',15)
legend('lris-setosa', 'lris-versicolor', 'lris-virginica', 'FontSize',10,'Location','southeast')
legend('boxoff')

%%

features_norm = reshape(features_norm, [50 3 4]);

figure(3)
for i = 1:4
    subplot(4,4,i)
    plot(features_norm(:,:,1), features_norm(:,:,i),'o','markersize',3)
    subplot(4,4,i+4)
    plot(features_norm(:,:,2), features_norm(:,:,i),'o','markersize',3)
    subplot(4,4,i+8)
    plot(features_norm(:,:,3), features_norm(:,:,i),'o','markersize',3)
    subplot(4,4,i+12)
    plot(features_norm(:,:,4), features_norm(:,:,i),'o','markersize',3)
end