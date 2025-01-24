clc;
clear;

load G4_RespAvg.mat
data = G4_RespAvg;
data_standardized = zscore(data);

[coeff, score, latent, tsquared, explained, mu] = pca(data_standardized);

disp('Variance explained by each principal component:');
disp(explained);

k = 2;
principal_components = score(:, 1:k);

figure;
scatter(principal_components(:, 1), principal_components(:, 2));
title('PCA of Neuron Responses');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;

figure;
pareto(explained);
title('Explained Variance by Principal Components');
xlabel('Principal Component');
ylabel('Variance Explained (%)');
