%% Numerical Verification of Accelerated MMD Variance Estimator
%  Goal: Validate the numerical consistency and decomposition of the proposed 
%        unbiased variance (euMMD) across different sample scales.
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."
clear; clc; close all;

% Parameter Settings
% Generate sample sizes using logarithmic spacing (from n=126 to n=15849)
sample_sizes = round(100 * 10.^([0.5, 1, 1.5, 2]));
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
    res1_gre = MMD_gretton(x, y,'laplace',beta);
    res1_pro = MMD_propose(x, y,'laplace',beta);
    % 2. Gaussian Kernel Benchmarking
    res2_gre = MMD_gretton(x, y,'gaussian',beta);
    res2_pro = MMD_propose(x, y,'gaussian',beta);

    % --- Structured Result Output ---
    fprintf('[Laplacian Kernel Result]\n');
    fprintf('  MMD Mean (Empirical):  %.3e\n', res1_pro.MMD2);      
    fprintf('  Gretton (Total Var):   %.3e\n', res1_gre.sigma2);   
    fprintf('  Proposed (Total Var):  %.3e\n', res1_pro.sigma2);   
    fprintf('  Proposed (2nd Var):    %.3e\n', res1_pro.sigma2_2); 
    fprintf('\n');
    fprintf('[Gaussian Kernel Result]\n');
    fprintf('  MMD Mean (Empirical):  %.3e\n', res2_pro.MMD2);      
    fprintf('  Gretton (Total Var):   %.3e\n', res2_gre.sigma2);   
    fprintf('  Proposed (Total Var):  %.3e\n', res2_pro.sigma2);   
    fprintf('  Proposed (2nd Var):    %.3e\n', res2_pro.sigma2_2); 
end
