%% Numerical Verification of Accelerated MMD Variance Estimation
%  Goal: Validate the scalability and numerical stability of euMMD under 
%        imbalanced sample settings (m = 2n).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% Parameter Settings
% Sample sizes distributed logarithmically to observe convergence rates
sample_sizes = round(100 * 10.^([0.5, 1, 1.5, 2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 0;             % Location shift (Null Hypothesis H0)

% Experimental Loop for Scaling Analysis
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = round(0.8*n);
    
    fprintf('\n');
    fprintf('Running Simulation: n = %d, m = %d (Total N = %d)\n', n, m, n+m);

    % Synthetic Data Generation (Laplace Distribution)
    % P ~ Laplace(0,1), Q ~ Laplace(0,1)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % Proposed euMMD Variance Computation
    % Estimates include total variance (sigma2) and its components
    res1 = MMD_propose(x, y,'laplace',beta); % Using Laplacian Kernel
    res2 = MMD_propose(x, y,'gaussian',beta);% Using Gaussian Kernel
   
    % Structured Result Output
    fprintf('  [Laplace Kernel]\n');
    fprintf('    MMD Mean (Empirical):       %.3e\n', res1.MMD2);
    fprintf('    Total Variance (sigma2):    %.3e\n', res1.sigma2);
    fprintf('    Second-order Var (sigma2_2): %.3e\n', res1.sigma2_2);
    
    fprintf('  [Gaussian Kernel]\n');
    fprintf('    MMD Mean (Empirical):       %.3e\n', res2.MMD2);
    fprintf('    Total Variance (sigma2):    %.3e\n', res2.sigma2);
    fprintf('    Second-order Var (sigma2_2): %.3e\n', res2.sigma2_2);
end
