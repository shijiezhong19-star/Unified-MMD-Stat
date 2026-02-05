%% MMD Space Complexity Benchmark: Accelerated euMMD Implementation
%  Goal: Measure the peak memory footprint of the O(N log N) algorithm (euMMD).
%  Method: Uses a high-frequency (1ms) timer-based monitor to capture the 
%          "Memory Watermark" during execution.
%  Key Feature: Implements rigorous environment cleanup to isolate the algorithm's 
%               incremental memory consumption from MATLAB's baseline.

clear; clc; close all;

% Configuration
num_trials = 20;
n_matrix = [200, 500, 1000, 2000, 5000, 10000]; 
n_fast_only = []; 
beta = 1.0;
all_n = [n_matrix, n_fast_only];
time_matrix = zeros(length(n_matrix), 1);
space_matrix = time_matrix;
time_proposed = zeros(length(all_n), 1);
space_proposed = time_proposed;

% === 4. 加速算法 - 空间测试 (独立脚本版) ===
global current_peak_mem;

% 1. Environment Sanitization
% Clear variables to ensure a clean heap before benchmarking
clearvars -except all_n beta num_trials; 
% Force Java Garbage Collection to release unused memory back to the JVM
java.lang.System.gc();
pause(3);

% 2. Re-calibrate Baseline "Zero" Memory
m_init = memory;
base_mem = m_init.MemUsedMATLAB; 
fprintf('\nEnvironment reset complete. Baseline Memory: %.2f MB\n', base_mem/1024^2);
fprintf('--- Accelerated Algorithm: Independent Space Profiling ---\n');

% 3. Initialize Memory Monitoring Timer
% ExecutionMode 'fixedRate' at 0.001s captures transient allocations
t_monitor = timer('ExecutionMode', 'fixedRate', ...
                 'Period', 0.001, ... 
                 'TimerFcn', @(~,~) monitorMemory());

space_proposed = zeros(size(all_n));

% 4. Benchmarking Loop
for i = 1:length(all_n)
    n = all_n(i);
    x = randn(n,1); y = 2*randn(1.2*n,1);
    
    current_peak_mem = 0;
    drawnow; % Flush event queue to ensure timer starts precisely
    start(t_monitor);
    
    % Execute Accelerated Algorithm (O(N) Space)
    euMMD(x, y, beta);
    
    m_snapshot = memory;
    stop(t_monitor);
    
    % Calculate Peak Incremental Memory Usage
    trial_peak = max(current_peak_mem, m_snapshot.MemUsedMATLAB);
    diff_val = trial_peak - base_mem;
    
    % Logic Correction: 
    % If diff_val is negative, the algorithm's footprint is smaller than 
    % system background fluctuations. We assign a nominal value (0.01MB) 
    % to prevent negative axis errors in log-scale plotting.
    if diff_val < 0
        space_proposed(i) = 0.01 * 1024^2;
    else
        space_proposed(i) = diff_val;
    end
    
    fprintf('n = %d: Peak Memory Delta = %.2f MB\n', n, space_proposed(i)/1024^2);
end

% Cleanup and Data Export
delete(t_monitor);
save('space_proposed.mat', 'space_proposed');

% Monitor Callback Function
% Captured within the global scope to track the peak "Watermark"
function monitorMemory()
    global current_peak_mem;
    m = memory;
    if m.MemUsedMATLAB > current_peak_mem
        current_peak_mem = m.MemUsedMATLAB;
    end
end