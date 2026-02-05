%% MMD Variance Efficiency Benchmark (Empirical Time Complexity)
%  Goal: Compare the execution time of the standard O(N^2) matrix implementation 
%        versus the proposed O(N log N) accelerated algorithm (euMMD).
%  Setting: Execution times are averaged over multiple trials to ensure stability.

clear; clc; close all;

% Configuration
num_trials = 20;
n_matrix = [200, 500, 1000, 2000, 5000, 10000]; 
n_fast_only = [20000, 50000, 100000, 500000, 1000000]; 
beta = 1.0;
all_n = [n_matrix, n_fast_only];

% Preallocate memory for timing results
time_matrix = zeros(length(n_matrix), 1);
time_proposed = zeros(length(all_n), 1);

% Stage 1: Benchmarking Standard Matrix Algorithm (O(N^2))
fprintf('Stage 1: Standard Matrix-form Estimator (O(N^2))\n');
for i = 1:length(n_matrix)
    n = n_matrix(i);
    t_acc = 0;
    for t = 1:num_trials
        x = randn(n,1); y = 2*randn(1.2*n,1);
        tic;
        MMD_propose(x, y,'laplace', beta); 
        t_acc = t_acc + toc;
    end
    time_matrix(i) = t_acc / num_trials;
    fprintf('n = %d: Average Time = %.4f s\n', n, time_matrix(i));
end

% Stage 2: Benchmarking Proposed Accelerated Algorithm (O(N log N))
fprintf('\nStage 2: Proposed euMMD Estimator (O(N log N))\n');
for i = 1:length(all_n)
    n = all_n(i);
    t_acc = 0;
    for t = 1:100
        x = randn(n,1); y = 2*randn(1.2*n,1);
        tic;
        euMMD(x, y, beta);
        t_acc = t_acc + toc;
    end
    time_proposed(i) = t_acc / num_trials;
    fprintf('n = %d: Average Time = %.4f s\n', n, time_proposed(i));
end

% Save Results for Plotting
save time_matrix time_matrix;
save time_proposed time_proposed;
