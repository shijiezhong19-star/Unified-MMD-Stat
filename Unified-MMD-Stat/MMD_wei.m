function result = MMD_wei(X, Y, kernel_type, param)
% MMD_WEI - Asymptotic MMD^2 and variance for unequal sample sizes.
%
% This function implements the MMD^2 estimator and its asymptotic variance 
% based on the generalized U-statistic framework described in:
% Wei, A., Jalali, M., & Sutherland, D. J. (2025). "Maximum Mean Discrepancy 
% with Unequal Sample Sizes via Generalized U-Statistics". arXiv:2512.13997.
%
% Input:
%   X, Y         : Sample matrices of size nX x d and nY x d
%   kernel_type  : 'gaussian' or 'laplace'
%   param        : Kernel bandwidth parameter
%
% Output:
%   result.MMD2   - Standard MMD^2 estimator
%   result.sigma2 - Asymptotic variance

    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same feature dimension.');
    end

    nX = size(X,1);
    nY = size(Y,1);
    % n is used as the reference scale for the asymptotic ratio
    n  = min(nX, nY);

    % Kernel Matrix Computation
    Kxx = kernel_func(X, X, kernel_type, param);
    Kyy = kernel_func(Y, Y, kernel_type, param);
    Kxy = kernel_func(X, Y, kernel_type, param);

    % MMD^2 Estimator
    % Based on the standard V-statistic or U-statistic representation
    MMD2 = mean(Kxx(:)) + mean(Kyy(:)) - 2*mean(Kxy(:));

    % Component-wise Conditional Expectations
    % These terms estimate the first-order Hoeffding decomposition components
    % EXX_i = E_{x'} [k(X_i, x')], etc.
    EXX = mean(Kxx, 2);
    EXY = mean(Kxy, 2);
    EYY = mean(Kyy, 2);
    EYX = mean(Kxy, 1)';

    % Zeta Calculation (Influence Functions)
    % zeta_X represents the variance contribution from the X-distribution
    var_EXX = mean(EXX.^2) - mean(EXX)^2;
    var_EXY = mean(EXY.^2) - mean(EXY)^2;
    cov_X   = mean(EXX .* EXY) - mean(EXX)*mean(EXY);
    zeta_X = var_EXX + var_EXY - 2*cov_X;

    % zeta_Y represents the variance contribution from the Y-distribution
    var_EYY = mean(EYY.^2) - mean(EYY)^2;
    var_EYX = mean(EYX.^2) - mean(EYX)^2;
    cov_Y   = mean(EYY .* EYX) - mean(EYY)*mean(EYX);
    zeta_Y = var_EYY + var_EYX - 2*cov_Y;

    % Asymptotic Variance Assembly
    % rho represents the relative sample weights as nX, nY -> infinity
    rho_X = n / nX;
    rho_Y = n / nY;

    % The asymptotic variance accounts for the imbalance between nX and nY
    sigma2 = (4*rho_X*zeta_X + 4*rho_Y*zeta_Y)/n;

    % Output Packaging
    result.MMD2   = MMD2;
    result.sigma2 = sigma2;
end

%% Vectorized Kernel Function
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