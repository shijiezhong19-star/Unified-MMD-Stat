%% Numerical Verification of Accelerated MMD Variance Estimation
%  Goal: Validate the scalability and numerical stability of euMMD under 
%        imbalanced sample settings (m = 2n).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% Parameter Settings
% Sample sizes distributed logarithmically to observe convergence rates
sample_sizes = round(100 * 10.^([0.1, 0.8, 1.5, 2.2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 0;             % Location shift (Null Hypothesis H0)

% Experimental Loop for Scaling Analysis
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = round(2*n);
    
    fprintf('\n');
    fprintf('Running Simulation: n = %d, m = %d (Total N = %d)\n', n, m, n+m);

    % Synthetic Data Generation (Laplace Distribution)
    % P ~ Laplace(0,1), Q ~ Laplace(0,1)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % Proposed euMMD Variance Computation
    % Estimates include total variance (sigma2) and its components
    res31 = MMD_propose(x, y,'laplace',beta); % Using Laplacian Kernel
    res32 = MMD_propose(x, y,'gaussian',beta);% Using Gaussian Kernel
   
    % Output Results
    % Monitoring the decay of total variance as sample size increases
    fprintf('  [Laplace Kernel] Total Variance (sigma2): %.3e\n', res31.sigma2);
    fprintf('  [Gaussian Kernel] Total Variance (sigma2): %.3e\n', res32.sigma2);
end
