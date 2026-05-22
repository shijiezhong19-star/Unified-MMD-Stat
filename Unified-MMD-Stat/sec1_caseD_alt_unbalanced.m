%% Numerical Verification of Accelerated MMD Variance Estimation (Imbalanced H1)
%  Goal: Validate the "Unified" euMMD framework under imbalanced settings (m=2n) 
%        and the Alternative Hypothesis (delta=1).
%  Reference: Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD..."

clear; clc; close all;

% --- Parameter Settings ---
% Sample sizes using logarithmic spacing to verify convergence across scales
sample_sizes = round(100 * 10.^([0.5, 1, 1.5, 2]));
beta = 1.0;            % Kernel bandwidth (sigma)
delta = 1;             % Location shift (P != Q, Alternative Hypothesis H1)

% Experimental Loop for Scaling & Stability Analysis
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); m = round(0.8*n);
    
    fprintf('\n');
    fprintf('Test Case %d: n = %d, m = %d (Imbalanced H1 Scenario)\n', idx, n, m);

    % Synthetic Data Generation (Laplace Distribution)
    % Generating samples from Laplace(0,1) and Laplace(1,1)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;

    % 1. Laplacian Kernel Benchmarking
    res1_wei = MMD_wei(x, y,'laplace',beta);
    res1_pro = MMD_propose(x, y,'laplace',beta);

    % 2. Gaussian Kernel Benchmarking
    res2_wei = MMD_wei(x, y,'gaussian',beta);
    res2_pro = MMD_propose(x, y,'gaussian',beta);
   
    % --- Structured Result Output ---
    fprintf('  [Laplace Kernel]\n');
    fprintf('    MMD Mean (Empirical):        %.3e\n', res1_pro.MMD2);      
    fprintf('    Wei et al.: %.3e\n', res1_wei.sigma2);   
    fprintf('    Proposed (First-order):    %.3e\n', res1_pro.sigma2_1);   
    fprintf('    Proposed (Second-order):  %.3e\n', res1_pro.sigma2_2); 
    fprintf('  [Gaussian Kernel]\n');
    fprintf('    MMD Mean (Empirical):        %.3e\n', res2_pro.MMD2);      
    fprintf('    Wei et al.: %.3e\n', res2_wei.sigma2);   
    fprintf('    Proposed (First-order):    %.3e\n', res2_pro.sigma2_1);   
    fprintf('    Proposed (Second-order):  %.3e\n', res2_pro.sigma2_2);
end
