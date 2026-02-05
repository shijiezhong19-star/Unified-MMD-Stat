%% Numerical Verification of Accelerated MMD Variance Estimation (Imbalanced H1)
%  Goal: Validate the "Unified" euMMD framework under imbalanced settings (m=2n) 
%        and the Alternative Hypothesis (delta=1).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% --- Parameter Settings ---
% Sample sizes using logarithmic spacing to verify convergence across scales
sample_sizes = round(100 * 10.^([0.1, 0.8, 1.5, 2.2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 1;             % Location shift (P != Q, Alternative Hypothesis H1)

% Experimental Loop for Scaling & Stability Analysis
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = round(2*n);
    
    fprintf('\n');
    fprintf('Test Case %d: n = %d, m = %d (Imbalanced H1 Scenario)\n', idx, n, m);

    % Synthetic Data Generation (Laplace Distribution)
    % Generating samples from Laplace(0,1) and Laplace(1,1)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % 1. Laplacian Kernel Benchmarking
    % Comparing Wei (Asymptotic/Linear) vs. Proposed (Unified/Accelerated)
    res21 = MMD_wei(x,y,'laplace',beta);
    res31 = MMD_propose(x, y,'laplace',beta);

    % 2. Gaussian Kernel Benchmarking
    res22 = MMD_wei(x,y,'gaussian',beta);
    res32 = MMD_propose(x, y,'gaussian',beta);
   
    % --- Structured Result Output ---
    % Note: Under H1, 'Pro' (Total Variance) is dominated by first-order terms.
    % 'Pro_2' highlights the exactness of the second-order component.
    fprintf('[Laplacian Kernel Result]\n');
    fprintf('  Wei: %.3e | Proposed (Total): %.3e | Proposed (T2): %.3e\n', ...
        res21.sigma2, res31.sigma2, res31.sigma2_2);
    
    fprintf('[Gaussian Kernel Result]\n');
    fprintf('  Wei: %.3e | Proposed (Total): %.3e | Proposed (T2): %.3e\n', ...
        res22.sigma2, res32.sigma2, res32.sigma2_2);
end
