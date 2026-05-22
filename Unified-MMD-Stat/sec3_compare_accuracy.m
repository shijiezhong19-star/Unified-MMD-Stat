%% Numerical Equivalence Verification: Matrix-form vs. Accelerated euMMD
%  Goal: Confirm that the accelerated O(N log N) algorithm (euMMD) yields 
%        identical results to the exact U-statistic formula across scales.
%  Setting: Imbalanced samples (m = 1.2n) under Alternative Hypothesis (H1).

clear; clc; close all;

% Parameter Settings
sample_sizes = [10, 100, 1000, 10000];
beta = 1.0;
delta = 1;

% Simulation Loop
for idx = 1:length(sample_sizes)
    n = sample_sizes(idx); 
    m = round(n * 1.2);
    
    fprintf('\n');
    fprintf('Simulating: n=%d, m=%d (Total N=%d)\n', n, m, n+m);
    
    % Synthetic Data Generation (Laplace Distribution)
    u1 = rand(n,1); x = -sign(u1-0.5).*log(1-2*abs(u1-0.5));
    u2 = rand(m,1); y = -sign(u2-0.5).*log(1-2*abs(u2-0.5)) + delta;
        
    % --- Algorithm Execution ---
    % res1: Standard matrix-form implementation (Ground Truth)
    % res2: Proposed accelerated prefix-suffix implementation (euMMD)
    res1 = MMD_propose(x, y,'laplace', beta);
    res2 = euMMD(x, y, beta);
        
    % --- Verification Output ---
    % Note: sigma_1 and sigma_2 refer to first- and second-order components.
    % After removing the '-n' logic error, these should match to machine precision.
    fprintf('  [Values]  MMD2: %.3e | euMMD: %.3e\n', res1.MMD2, res2.MMD2);
    fprintf('  [Total Var]  Matrix_Sig: %.3e | euMMD_Sig: %.3e\n', res1.sigma2, res2.sigma2);
    fprintf('  [Comp 1] Matrix_Sig1: %.3e | euMMD_Sig1: %.3e\n', res1.sigma2_1, res2.sigma2_1);
    fprintf('  [Comp 2] Matrix_Sig2: %.3e | euMMD_Sig2: %.3e\n', res1.sigma2_2, res2.sigma2_2);
end
