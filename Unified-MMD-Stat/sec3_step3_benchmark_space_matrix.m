%% MMD Space Complexity Benchmark (Memory Usage Profiling)
%  Goal: Measure the peak memory consumption (Memory Watermark) of the 
%        standard O(N^2) matrix implementation as a function of sample size.
%  Method: Uses a high-frequency timer (1ms) to capture transient peak memory 
%          allocations during the kernel matrix construction.

clear; clc; close all;

% Configuration
num_trials = 20;
n_matrix = [200, 500, 1000, 2000, 5000, 10000]; 
n_fast_only = []; 
beta = 1.0;
all_n = [n_matrix, n_fast_only];

% Preallocate arrays for space complexity results
time_matrix = zeros(length(n_matrix), 1);
space_matrix = time_matrix;
time_proposed = zeros(length(all_n), 1);
space_proposed = time_proposed;

global current_peak_mem;

% Baseline Memory Setup
% Capture the initial memory footprint of MATLAB to calculate the incremental delta
m_init = memory;
base_mem = m_init.MemUsedMATLAB;

% Define High-Frequency Memory Monitor
% Period of 0.001s is the physical limit for MATLAB timers to ensure 
% detection of transient matrix allocations.
t_monitor = timer('ExecutionMode', 'fixedRate', ...
                 'Period', 0.001, ... 
                 'TimerFcn', @(~,~) monitorMemory());


% Benchmarking Standard Matrix Algorithm (O(N^2) Space)
fprintf('\nO(N^2) Matrix Implementation: Space Profiling\n');
for i = 1:length(n_matrix)
    n = n_matrix(i);
    
    % Prepare synthetic data
    x = randn(n,1); y = 2*randn(1.2*n,1);
    
    % Reset peak tracking for current sample scale
    current_peak_mem = 0;
    
    % Ensure command queue is clear before starting the monitor
    drawnow; 
    start(t_monitor);
    
    % Execute the algorithm (Heavy memory allocation occurs here)
    MMD_propose(x, y,'laplace', beta); 
    
    % Capture immediate snapshot after execution and halt monitor
    m_snapshot = memory; 
    stop(t_monitor);
    
    % Calculate incremental Peak Memory Usage (Delta from baseline)
    % Takes the maximum of either the timer captures or the post-execution state
    trial_peak = max(current_peak_mem, m_snapshot.MemUsedMATLAB);
    space_matrix(i) = trial_peak - base_mem;
    
    fprintf('n = %d: Peak Memory Delta = %.2f MB\n', n, space_matrix(i)/1024^2);
end

delete(t_monitor);

save space_matrix space_matrix;

% Monitor Callback Function
% This function is triggered every 1ms to catch peak memory usage that 
% may occur briefly during large matrix inversions or multiplications.
function monitorMemory()
    global current_peak_mem;
    m = memory;
    if m.MemUsedMATLAB > current_peak_mem
        current_peak_mem = m.MemUsedMATLAB;
    end
end