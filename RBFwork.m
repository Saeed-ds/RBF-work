clc;
clear;
close all;

%% ================= USER-DEFINED PARAMETERS =================
numSamples = input('Enter the number of samples: ');
M = input('Enter number of RBF neurons (M): ');
useBias = input('Use bias term? (1 = yes, 0 = no): ');
autoSigma = input('Use automatic sigma? (1 = yes, 0 = no): ');
lambda = input('Enter regularization parameter (lambda): ');

if autoSigma == 0
    manualSigma = input('Enter sigma value: ');
end

%% ================= DATA =================
X = linspace(-pi, pi, numSamples)';
T = sin(X) .* cos(X);

N = length(X);   % actual number of samples

%% ================= RBF CENTERS =================
centers = linspace(min(X), max(X), M)';

%% ================= SIGMA =================
if autoSigma
    d = diff(centers);
    sigma = mean(d);
else
    sigma = manualSigma;
end

%% ================= DESIGN MATRIX =================
Phi = zeros(N, M);
for i = 1:M
    Phi(:, i) = exp(-((X - centers(i)).^2) / (2 * sigma^2));
end

if useBias
    Phi = [Phi ones(N, 1)];
end

%% ================= TRAINING (RIDGE REGRESSION) =================
I = eye(size(Phi, 2));
W = (Phi' * Phi + lambda * I) \ (Phi' * T);

%% ================= OUTPUT =================
Y = Phi * W;
MSE = mean((T - Y).^2);

%% ================= DISPLAY RBF INFORMATION =================
fprintf('\n============================================\n');
fprintf('      RBF NEURAL NETWORK INFORMATION\n');
fprintf('============================================\n');
fprintf('Number of samples        : %d\n', N);
fprintf('Hidden neurons (M)       : %d\n', M);
fprintf('Sigma (σ)                : %.6f\n', sigma);
fprintf('Regularization (lambda)  : %.6e\n', lambda);
fprintf('Bias enabled             : %s\n', string(logical(useBias)));
fprintf('Training method          : Least Squares (Ridge)\n');
fprintf('--------------------------------------------\n');
fprintf('Mean Squared Error (MSE) : %.8f\n', MSE);
fprintf('============================================\n\n');

%% ================= PLOT =================
figure;
plot(X, T, 'b', 'LineWidth', 2); hold on;
plot(X, Y, 'r--', 'LineWidth', 2);
legend('True Function', 'RBF Output');
title('RBF Neural Network Function Approximation');
grid on;