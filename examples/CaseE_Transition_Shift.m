%% Experiment: Variance Component Evolution & Sample Imbalance Invariance (Optimized)
clear; clc; close all;

% Configuration
n_fixed = 1000; 
m_list = [800, 1500, 2000]; 
sigma = 1.0;
delta_range = [0, logspace(-3, 0, 100)];

linestyle = {'-','-.','--'};

figure

for k = 1:length(m_list)
    m = m_list(k);
    % Fix sample pool to isolate the effect of distribution shift (delta)
    x = randn(n_fixed, 1);
    y0 = randn(m, 1);

    v_total = zeros(size(delta_range));
    v_2 = zeros(size(delta_range));

    for i = 1:length(delta_range)
        delta = delta_range(i);
        y = y0 + delta;
        % MMD_propose: Custom function for U-statistic decomposition
        res = MMD_propose(x, y, 'laplace', sigma);
        v_total(i) = res.sigma2;
        v_2(i) = res.sigma2_2;
    end

    % Subplot (a): Total Variance Dynamics
    subplot(2, 1, 1);
    hold on;
    plot(delta_range, v_total, linestyle{k}, ...
    'DisplayName', sprintf('$n=%d,\\; m=%d$', n_fixed, m));

    % Subplot (b): Second-order Component Stability
    subplot(2, 1, 2);
    hold on;
    plot(delta_range, v_2, linestyle{k}, ...
    'DisplayName', sprintf('$n=%d,\\; m=%d$', n_fixed, m));
end

%% Final Formatting and Theoretical Remarks
subplot(2, 1, 1);
title('$\mathrm{(a)\ Growth\  trend\  of\  the\  total\  variance}$', 'Interpreter', 'latex');
xlabel('$\mathrm{distribution\ shift\ \delta}$',  'Interpreter', 'latex');
ylabel('$\mathrm{total\ estimated\ variance}$',  'Interpreter', 'latex');

xlim([0, 1]);
legend('show');
leg = legend('show');
set(leg,  'Interpreter', 'latex');
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

subplot(2, 1, 2);
title('$\mathrm{(b)\ Growth\ trend\  of\  the\  second-order\  term\  variance}$', 'Interpreter', 'latex');
xlabel('$\mathrm{distribution\ shift\ \delta}$',  'Interpreter', 'latex');
ylabel('$\mathrm{second-order term\ estimated\ variance}$',  'Interpreter', 'latex');

xlim([0, 1]);
legend('show');
leg = legend('show');
set(leg,  'Interpreter', 'latex');
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')