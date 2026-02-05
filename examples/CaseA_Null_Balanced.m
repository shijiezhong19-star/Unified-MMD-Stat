%% Numerical Verification of Accelerated MMD Variance Estimator
%  Goal: Validate the numerical consistency and decomposition of the proposed 
%        unbiased variance (euMMD) across different sample scales.
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."
clear; clc; close all;

% Parameter Settings
% Generate sample sizes using logarithmic spacing (from n=126 to n=15849)
sample_sizes = round(100 * 10.^([0.1, 0.8, 1.5, 2.2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 0;             % Location shift (set to 0 for Null Hypothesis H0)

% Experimental Loop
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = n;

    fprintf('\n');
    fprintf('Test Case %d: Sample Size n = m = %d\n', idx, n);

    % Data Generation (P ~ Laplace(0,1), Q ~ Laplace(0,1))
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % 1. Laplacian Kernel Benchmarking
    % Standard asymptotic variance (Gretton), bias-corrected (Sutherland), and euMMD (Proposed)
    res11 = MMD_gretton(x,y,'laplace',beta);
    res21 = MMD_sutherland(x,y,'laplace',beta);
    res31 = MMD_propose(x, y,'laplace',beta);

    % 2. Gaussian Kernel Benchmarking
    res12 = MMD_gretton(x,y,'gaussian',beta);
    res22 = MMD_sutherland(x,y,'gaussian',beta);
    res32 = MMD_propose(x, y,'gaussian',beta);

    % --- Structured Result Output ---
    % Note: Under H0, 'Proposed (T2)' should align closely with 'Gretton'
    fprintf('[Laplacian Kernel Result]\n');
    fprintf('  Gretton (H0-Asymp): %.3e | Sutherland (Unbiased): %.3e\n', res11.sigma2, res21.sigma2);
    fprintf('  Proposed (Total):   %.3e | Proposed (T2 Only):   %.3e\n', res31.sigma2, res31.sigma2_2);
    
    fprintf('[Gaussian Kernel Result]\n');
    fprintf('  Gretton (H0-Asymp): %.3e | Sutherland (Unbiased): %.3e\n', res12.sigma2, res22.sigma2);
    fprintf('  Proposed (Total):   %.3e | Proposed (T2 Only):   %.3e\n', res32.sigma2, res32.sigma2_2);
end
