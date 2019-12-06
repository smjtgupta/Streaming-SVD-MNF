close all;
clc;
clear;

addpath('data', 'helper')

load 'target_062_high'

A = reshape(C, [], 754);

Ahat = sketchyMNF(A, 150);

AA = MNF(A);

%% stats

figure;
plot(A(200,:), 'r');
hold on;
plot(Ahat(200,:)+0.1, 'g');
hold on;
plot(AA(200,:)+0.2, 'b');
title('Denoising Plots')
xlabel('Channel')
legend('Noisy', 'SketchyMNF', 'MNF', 'location', 'northeast');