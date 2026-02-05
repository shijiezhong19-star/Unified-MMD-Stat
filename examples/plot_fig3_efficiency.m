clear; clc; close all;

n_matrix = [200, 500, 1000, 2000, 5000, 10000]; 
n_fast_only = [20000, 50000, 100000, 500000, 1000000]; 
all_n = [n_matrix, n_fast_only];

load("time_matrix.mat");
load("time_proposed.mat");
load("space_matrix.mat");
load("space_proposed.mat");

figure
subplot(2,1,1)
hold on;
loglog(n_matrix(:), time_matrix(:), 'o-', 'DisplayName', '$\mathrm{MMD}$');
loglog(all_n(:), time_proposed(:), 's--', 'DisplayName', '$\mathrm{euMMD}$');
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('$\mathrm{sample \, size}\,n$',  'Interpreter', 'latex');
ylabel('$\mathrm{runtime\,, s}$',  'Interpreter', 'latex');
title('$\mathrm{(a)\, Comparison\, of\, runtime}$', 'Interpreter', 'latex');
xlim([200, 1000000]);
legend('show');
leg = legend('show');
set(leg,  'Interpreter', 'latex' ...
    , 'FontName', 'Times New Roman', 'FontSize', 9, 'Box', 'on', 'Location', 'northwest');

subplot(2,1,2)
hold on;
loglog(n_matrix(:), abs(space_matrix(:))/1024^2, 'o-', 'DisplayName', '$\mathrm{MMD}$');
loglog(n_matrix(:), abs(space_proposed(:))/1024^2, 's--', 'DisplayName', '$\mathrm{euMMD}$');
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('$\mathrm{sample \, size}\,n$',  'Interpreter', 'latex');
ylabel('$\mathrm{memory\,, MB}$',  'Interpreter', 'latex');
title('$\mathrm{(b)\, Comparison \,of \,memory \,usage}$', 'Interpreter', 'latex');
xlim([200, 10000]);
legend('show');
leg = legend('show');
set(leg,  'Interpreter', 'latex' ...
    , 'FontName', 'Times New Roman', 'FontSize', 9, 'Box', 'on', 'Location', 'northwest');