function result = MMD_sutherland(X, Y, kernel_type, param)

% MMD_SUTHERLAND - Quadratic-time unbiased MMD^2 and its variance estimator.
%
% This function implements the unbiased variance estimator for MMD^2 as
% proposed in:
% Sutherland, D. J., & Deka, N. (2019). "Unbiased estimators for the 
% variance of MMD estimators". arXiv preprint arXiv:1906.02104.
%
% Input:
%   X, Y         : Sample matrices of size m x d and n x d
%   kernel_type  : 'gaussian' or 'laplace'
%   param        : Kernel bandwidth
%
% Output:
%   result.MMD2   - Unbiased MMD^2 estimator
%   result.sigma2 - Unbiased estimator of the variance of the MMD^2
%
% Note: This implementation follows the matrix-based exact computation 
% of the U-statistic variance components.

    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same feature dimension.');
    end

    m = size(X,1);% Number of samples in X
    n = size(Y,1);% Number of samples in Y

    % Kernel Matrix Computation
    Kxx = kernel_func(X, X, kernel_type, param);
    Kyy = kernel_func(Y, Y, kernel_type, param);
    Kxy = kernel_func(X, Y, kernel_type, param);

    % Pre-calculating tilde(K) by removing self-similarities (diagonal)
    % This is crucial for unbiased U-statistics
    Kxx(1:m+1:end) = 0;
    Kyy(1:n+1:end) = 0;

    % Unbiased MMD^2 (Quadratic-time)
    MMD2 = sum(Kxx(:))/(m*(m-1)) ...
            + sum(Kyy(:))/(n*(n-1)) ...
            - 2*sum(Kxy(:))/(m*n);

    % Matrix-based Variance Components Precomputation
    % Efficiently compute row sums and quadratic forms
    one_m = ones(m,1);
    one_n = ones(n,1);
    Kxx1 = Kxx * one_m;
    Kyy1 = Kyy * one_n;
    Kxy1 = Kxy * one_n;
    Kyx1 = Kxy' * one_m;

    % Norms and quadratic forms
    normKxx1_2 = sum(Kxx1.^2);
    normKyy1_2 = sum(Kyy1.^2);
    normKxxF_2 = sum(Kxx(:).^2);
    normKyyF_2 = sum(Kyy(:).^2);
    normKxyF_2 = sum(Kxy(:).^2);

    % Triple product terms: 1' * K * K * 1
    oneKxx1 = one_m' * Kxx1;
    oneKyy1 = one_n' * Kyy1;
    oneKxy1 = one_m' * Kxy1;
    mixXX = one_m' * (Kxx * Kxy1);
    mixYY = one_n' * (Kyy * Kyx1);

    % Variance Estimator (Vm) Assembly
    % The following coefficients represent the complex combinatorial terms 
    % derived for the unbiased variance of a U-statistic.

    m2 = m*(m-1);
    n3 = n*(n-1)*(n-2);
    n4 = n*(n-1)*(n-2)*(n-3);

    Vm = 0;
    Vm = Vm + 4*(m*n + m - 2*n)/(m2*n4) * ...
        (normKxx1_2 + normKyy1_2);
    Vm = Vm - 2*(2*m - n)/(m*n*(m-1)*(n-2)*(n-3)) * ...
        (normKxxF_2 + normKyyF_2);
    Vm = Vm + 4*(m*n + m - 2*n - 1)/(m2*n^2*(n-1)^2) * ...
        (sum(Kxy1.^2) + sum(Kyx1.^2));
    Vm = Vm - 4*(2*m - n - 2)/(m2*n*(n-1)^2) * normKxyF_2;
    Vm = Vm - 2*(2*m - 3)/(m2*n4) * ...
        (oneKxx1^2 + oneKyy1^2);
    Vm = Vm - 4*(2*m - 3)/(m2*n^2*(n-1)^2) * oneKxy1^2;
    Vm = Vm - 8/(m*n3) * (mixXX + mixYY);
    Vm = Vm + 8/(m*n*n3) * ...
        (oneKxx1 + oneKyy1) * oneKxy1;

    % Results
    result.MMD2 = MMD2;
    result.sigma2 = Vm;
end

%% Kernel Function (Vectorized)
function K = kernel_func(X, Y, type, param)
    switch lower(type)
        case 'gaussian'
            XX = sum(X.^2,2);
            YY = sum(Y.^2,2);
            K = exp(-(XX + YY' - 2*(X*Y'))/(2*param^2));

        case 'laplace'
            XX = sum(X.^2,2);
            YY = sum(Y.^2,2);
            D = sqrt(max(XX + YY' - 2*(X*Y'),0));
            K = exp(-D/param);
    end
end