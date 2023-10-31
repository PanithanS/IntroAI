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

%% k-means clustering

% Set the number of clusters (k)
k = 3;

% Perform k-means clustering
[idx, C] = kmeans(features_norm, k);
% 'idx' contains the cluster assignment for each data point
% 'C' contains the centroid coordinates for each cluster
%%
% Display the cluster assignments
disp('Cluster Assignments:');
disp(idx);

% Display the centroid coordinates
disp('Centroid Coordinates:');
disp(C);

% Plot the data points with different colors for each cluster
figure;
gscatter(features_norm(:, 1), features_norm(:, 2), idx);
title('K-Means Clustering');