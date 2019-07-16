close all;
clc;
clear;

load('test.mat')
A = rgb2gray(A); %% input data matrix

%% Full SVD

[U,S,V] = svd(A);

figure;
imshow(A);
title('Original Image')

%% Sketchy SVD (emulating streaming setting) 

r = 350; % estimated rank
alpha = 0; % overstimate parameter
k = r + alpha; % overestimated rank

[sketch_U, sketch_S, sketch_V] = sketchy_svd(A, r, k);

AA = sketch_U * sketch_S * sketch_V';
figure;
imshow(AA);
title('Approximated Image')

%% Stats

err_F = ['Frobenius Norm Error in Approximation: ', num2str(norm(A-AA,'fro'))];
disp(err_F);
err_2 = ['Spectral Norm Error in Approximation: ', num2str(norm(A-AA,2))];
disp(err_2);

figure;
plot(log(diag(S)),'r');
hold on;
plot(log(diag(sketch_S)), 'b');
hold on;
plot(zeros(size(S,1)),'g');
title('Spectral Decay')
xlabel('Features')
ylabel('Strength (log)')
legend('Original Spectrum', 'Approximated Spectrum', 'Location', 'northeast');