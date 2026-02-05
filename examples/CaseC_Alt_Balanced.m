%% Numerical Verification of Accelerated MMD Variance Estimation (H1 Scenario)
%  Goal: Evaluate variance estimator consistency and decomposition under the 
%        Alternative Hypothesis (H1) with a location shift (delta = 1).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% Parameter Settings
% Logarithmic spacing for sample sizes to analyze O(n^-1) convergence
sample_sizes = round(100 * 10.^([0.1, 0.8, 1.5, 2.2]));
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
    res11 = MMD_sutherland(x,y,'laplace',beta);
    res21 = MMD_wei(x,y,'laplace',beta);
    res31 = MMD_propose(x, y,'laplace',beta);

    % 2. Gaussian Kernel Benchmarking
    res12 = MMD_sutherland(x,y,'gaussian',beta);
    res22 = MMD_wei(x,y,'gaussian',beta);
    res32 = MMD_propose(x, y,'gaussian',beta);

    % --- Structured Result Output ---
    % Under H1, Total Variance is dominated by the First-order term (T1)
    fprintf('[Laplacian Kernel Result]\n');
    fprintf('  Sutherland: %.3e | Wei: %.3e\n', res11.sigma2, res21.sigma2);
    fprintf('  Proposed (Total): %.3e | Proposed (T2 Component): %.3e\n', res31.sigma2, res31.sigma2_2);
    
    fprintf('[Gaussian Kernel Result]\n');
    fprintf('  Sutherland: %.3e | Wei: %.3e\n', res12.sigma2, res22.sigma2);
    fprintf('  Proposed (Total): %.3e | Proposed (T2 Component): %.3e\n', res32.sigma2, res32.sigma2_2);
end
