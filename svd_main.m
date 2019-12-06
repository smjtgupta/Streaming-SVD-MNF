close all;
clc;
clear;

addpath('data', 'helper')

% load('test.mat')
% A = rgb2gray(A); %% input data matrix

A = double(imread('lena512.bmp'));

r = 100; % estimate of rank

%% Full SVD

[U,S,V] = svd(A);

figure;
imshow(A, []);
title('Original Image')

%% Sketchy SVD 

Ahat = sketchySVD(A, r);

figure;
imshow(Ahat, []);
title('Approximated Image')

[sketch_U, sketch_S, sketch_V] = svd(Ahat);

%% Stats

err_F = ['Frobenius Norm Error in Approximation: ', num2str(norm(A-AA,'fro'))];
disp(err_F);
err_2 = ['Spectral Norm Error in Approximation: ', num2str(norm(A-AA,2))];
disp(err_2);

figure;
plot(log(diag(S)),'r');
hold on;
plot(log(diag(sketch_S)), 'b');
title('Spectral Decay')
xlabel('Features')
ylabel('Strength (log)')
legend('Original Spectrum', 'Approximated Spectrum', 'Location', 'northeast');