clear; clc; close all;
% Sample size ratios (m/n)
m_ratios = [0.5, 1.0, 1.5]; 
% Grid of sample sizes n from 10^2 to 10^4
n_steps = round(logspace(2, 4, 10)); 
% Maximum pool size for data generation
N_max = 20000; 
% Bandwidth for Gaussian kernel
sigma = 1.0;

% Data pool for distribution P
X_pool = randn(N_max, 1);
% Data pool for Q under Null Hypothesis (P = Q)
Y_pool_null = randn(N_max, 1);        
% Data pool for Q under Alternative Hypothesis (P != Q)
Y_pool_alt = randn(N_max, 1) + 1.5;   
% Structure to iterate through both hypothesis regimes
results = struct('name', {'(a)\ null\ hypothesis', '(b)\ alternative\ hypothesis'}, ...
                 'Y_pool', {Y_pool_null, Y_pool_alt});

figure;

for s = 1:2
    subplot(2, 1, s);
    hold on;
    
    for k = 1:length(m_ratios)
        ratio = m_ratios(k);
        v1_target = zeros(length(n_steps), 1);
        v2_target = zeros(length(n_steps), 1);
        
        for i = 1:length(n_steps)
            n = n_steps(i); 
            m = round(n * ratio);
            % Compute MMD and its variance components using the proposed estimator
            res = MMD_propose(X_pool(1:n), results(s).Y_pool(1:m), 'gaussian', sigma);
            
            if s == 1
                % Under Null: only extract second-order residual variance (V2)
                v1_target(i) = res.sigma2_2;
            else
                % Under Alternative: extract both second-order (V2) and first-order (V1)
                v1_target(i) = res.sigma2_2;
                v2_target(i) = res.sigma2_1;
            end
        end
        
        if s == 1
            loglog(n_steps, v1_target, ...
                'DisplayName', sprintf('$m/n=%.1f$ ($V_2$)', ratio));
        else
            loglog(n_steps, v1_target, ...
                'DisplayName', sprintf('$m/n=%.1f$ ($V_2$)', ratio));
            loglog(n_steps, v2_target, ...
                'DisplayName', sprintf('$m/n=%.1f$ ($V_1$)', ratio));
        end
    end
    grid on;
    title(['$\mathrm{', results(s).name, '}$'], 'Interpreter', 'latex');    
    xlabel('$\mathrm{sample\ size\ } n$', 'Interpreter', 'latex');
    
    if s == 1
        ylabel('$\mathrm{variance\ } V_2$', 'Interpreter', 'latex');
    else
        ylabel('$\mathrm{variance\ components}$', 'Interpreter', 'latex');
    end
    
    set(gca, 'XScale', 'log', 'YScale', 'log');
    leg=legend('show', 'Interpreter', 'latex', 'Location', 'southwest');
end
