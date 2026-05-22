clear; clc; close all;

% Base sample size
n = 1000;             
% Varying secondary sample sizes to evaluate imbalance
m_values = [500, 700, 2000, 3000];
% Number of Monte Carlo simulation trials
K = 1000;
% Bandwidth parameter for the Gaussian kernel
sigma = 1.0;
% Distribution mean shift under Alternative Hypothesis
delta = 1.5;

figure;

titles = {'(a) $m=500$ ($m/n=0.5$)';
          '(b) $m=700$ ($m/n=0.7$)';
          '(c) $m=2000$ ($m/n=2.0$)';
          '(d) $m=3000$ ($m/n=3.0$)'};

for s = 1:4
    m = m_values(s);
    fprintf('Running simulation for m = %d (%d/%d)...\n', m, s, length(m_values));
    % Preallocate memory for statistical properties
    mmd2_list = zeros(K, 1);
    nup_list  = zeros(K, 1);
    nuq_list  = zeros(K, 1);
    
    % Independent trials execution
    for k = 1:K
        % Generate data realizations from alternative regime
        X = randn(n, 1);
        Y = randn(m, 1) + delta; 
        % Evaluate the statistic and extract first-order variance projections
        res = MMD_propose(X, Y, 'gaussian', sigma);
        
        mmd2_list(k) = res.MMD2;
        nup_list(k)  = res.nuP;
        nuq_list(k)  = res.nuQ;
    end
    
    mean_mmd2 = mean(mmd2_list); 
    mean_nup = mean(nup_list);
    mean_nuq = mean(nuq_list);
    
    % Compute the sample limit ratios according to the theoretical framework
    N_min = min(n, m);
    rho_X = N_min / n;
    rho_Y = N_min / m;
    % Calculate limiting asymptotic variance (V_asy) and finite-sample standard deviation
    V_asy = 4 * rho_X * mean_nup + 4 * rho_Y * mean_nuq;

    var_theo = V_asy / N_min;
    std_theo = sqrt(var_theo);

    subplot(2, 2, s);
    hold on;
    
    histogram(mmd2_list, 30, 'Normalization', 'pdf', ...
        'DisplayName', '$\mathrm{Empirical\ Histogram}$');

    x_grid = linspace(mean_mmd2 - 4*std_theo, mean_mmd2 + 4*std_theo, 200);
    pdf_theo = normpdf(x_grid, mean_mmd2, std_theo);
    
    plot(x_grid, pdf_theo, ...
        'DisplayName', '$\mathrm{Theoretical\ } \mathcal{N}$');

    grid on;
    title(titles{s}, 'Interpreter', 'latex');
    xlabel('$\widehat{\mathrm{MMD}}^2$', 'Interpreter', 'latex');
    ylabel('$\mathrm{density}$', 'Interpreter', 'latex');
    
    xlim([mean_mmd2 - 3*std_theo, mean_mmd2 + 3*std_theo]);
    axis tight; 
end