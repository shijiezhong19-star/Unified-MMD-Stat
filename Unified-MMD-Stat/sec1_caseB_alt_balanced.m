%% Numerical Verification of Accelerated MMD Variance Estimation (H1 Scenario)
%  Goal: Evaluate variance estimator consistency and decomposition under the 
%        Alternative Hypothesis (H1) with a location shift (delta = 1).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% Parameter Settings
% Logarithmic spacing for sample sizes to analyze O(n^-1) convergence
sample_sizes = round(100 * 10.^([0.5, 1, 1.5, 2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 1;             % Location shift (P != Q, Alternative Hypothesis H1)

% Experimental Loop
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = n;
    
    fprintf('\n');
    fprintf('Simulation Case %d: n = m = %d (Alternative Hypothesis H1)\n', idx, n);

    % Synthetic Data Generation (Laplace Distribution)
    % P ~ Laplace(0,1), Q ~ Laplace(1,1)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % 1. Laplacian Kernel Benchmarking
    % Comparing Sutherland, Wei (Asymptotic/Linear), and Proposed (euMMD)
    res1_sur = MMD_sutherland(x, y,'laplace',beta);
    res1_pro = MMD_propose(x, y,'laplace',beta);

    % 2. Gaussian Kernel Benchmarking
    res2_sur = MMD_sutherland(x, y,'gaussian',beta);
    res2_pro = MMD_propose(x, y,'gaussian',beta);

    % --- Structured Result Output ---
    lap_abs_diff = abs(res1_pro.sigma2 - res1_sur.sigma2);
    fprintf('[Laplacian Kernel Result]\n');
    fprintf('  MMD Mean (Empirical):     %.3e\n', res1_pro.MMD2);      
    fprintf('  Sutherland (Total Var):   %.3e\n', res1_sur.sigma2);   
    fprintf('  Proposed (Total Var):     %.3e\n', res1_pro.sigma2);   
    fprintf('  Abs Variance Diff:        %.3e\n', lap_abs_diff);
    fprintf('\n');
    gau_abs_diff = abs(res2_pro.sigma2 - res2_sur.sigma2);
    fprintf('[Gaussian Kernel Result]\n');
    fprintf('  MMD Mean (Empirical):     %.3e\n', res2_pro.MMD2);      
    fprintf('  Sutherland (Total Var):   %.3e\n', res2_sur.sigma2);   
    fprintf('  Proposed (Total Var):     %.3e\n', res2_pro.sigma2);   
    fprintf('  Abs Variance Diff:        %.3e\n', gau_abs_diff);
end
