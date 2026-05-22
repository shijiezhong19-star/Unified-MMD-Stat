function result = MMD_gretton(X, Y, kernel_type, param)
% MMD_GRETTON implementation of unbiased MMD^2 and its variance estimator.
%
% This function implements the kernel two-sample test statistic and its 
% finite-sample variance under the null hypothesis, as described in:
% Gretton, A., et al. (2012). "A kernel two-sample test". JMLR.
%
% Input:
%   X, Y         : Sample matrices of size m x d
%   kernel_type  : 'gaussian' or 'laplace'
%   param        : Kernel bandwidth (sigma for Gaussian, b for Laplace)
%
% Output:
%   result.MMD2   - Unbiased MMD^2 estimator
%   result.sigma2 - Variance estimator under the Null Hypothesis (H0)

    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same feature dimension.');
    end
    m = size(X,1);

    % Kernel Matrix Computation
    % Compute pairwise kernel distance matrices
    Kxx = kernel_func(X, X, kernel_type, param);
    Kyy = kernel_func(Y, Y, kernel_type, param);
    Kxy = kernel_func(X, Y, kernel_type, param);

    % Unbiased MMD^2 Estimation
    % Standard U-statistic requires i ~= j for unbiasedness
    Kxx(1:m+1:end) = 0;
    Kyy(1:m+1:end) = 0;

    % Expectation terms for k(x,x'), k(y,y') and k(x,y)
    term_xx = sum(Kxx(:)) / (m*(m-1));
    term_yy = sum(Kyy(:)) / (m*(m-1));
    term_xy = sum(Kxy(:)) / (m^2);

    % Final unbiased MMD^2 formula
    MMD2 = term_xx + term_yy - 2*term_xy;

    % Variance sigma^2 Estimation (Under H0)
    % Construct the H-statistic core for U-statistic variance estimation
    H = Kxx + Kyy - Kxy - Kxy';
    H(1:m+1:end) = 0; % Eliminate diagonal terms

    % Extract upper triangular elements to compute variance of the U-statistic
    Hij = H(triu(true(m),1));
    Eh2 = mean(Hij.^2);

    % Under H0: P=Q, the variance of MMD2 is estimated as below:
    sigma2 = 2/(m*(m-1)) * (Eh2);
    
    % Result Packaging
    result.MMD2 = MMD2;
    result.sigma2 = sigma2;
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
