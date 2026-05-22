function result = MMD_propose(X, Y, kernel_type, param)
% MMD_PROPOSE - Unified finite-sample unbiased MMD^2 and variance estimator.
%
% This function implements the exact, algebraically strictly unbiased
% estimators for MMD variance components according to the latest derivations.
%
% Reference:
% Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for Maximum 
% Mean Discrepancy: Robust Finite-Sample Performance with Imbalanced Data..."
%
% Input:
%   X, Y         : Sample matrices of size n x d and m x d
%   kernel_type  : 'gaussian' or 'laplace'
%   param        : Kernel bandwidth parameter
%
% Output:
%   result.MMD2     - Unbiased MMD^2 estimate
%   result.sigma2   - Total unbiased variance (V1 + V2)
%   result.sigma2_1 - First-order component (V1)
%   result.sigma2_2 - Second-order component (V2)
%   result.rho1     - Ratio of first-order variance to total variance

    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same feature dimension.');
    end
    n = size(X,1);
    m = size(Y,1);
    
    if n < 4 || m < 4
        error('Sample sizes must be at least 4 for unbiased 4th-order variance estimation.');
    end

    % 1. Kernel Matrix Computation
    Kxx = kernel_func(X, X, kernel_type, param);
    Kyy = kernel_func(Y, Y, kernel_type, param);
    Kxy = kernel_func(X, Y, kernel_type, param);

    % Remove diagonal elements (within-sample) to satisfy i ~= j condition
    Kxx(1:n+1:end) = 0;
    Kyy(1:m+1:end) = 0;

    % 2. Unbiased MMD^2 Estimation
    MMD2 = sum(Kxx(:))/(n*(n-1)) + sum(Kyy(:))/(m*(m-1)) - 2*sum(Kxy(:))/(n*m);

    % 3. Precompute Row Sums, Norms and Projections
    % Base row/column sums
    Kxx1 = Kxx * ones(n, 1);   % n x 1
    Kyy1 = Kyy * ones(m, 1);   % m x 1
    Kxy1 = Kxy * ones(m, 1);   % n x 1
    Kyx1 = Kxy' * ones(n, 1);  % m x 1

    % Scalar sums (1^T K 1)
    sumKxx = sum(Kxx1);
    sumKyy = sum(Kyy1);
    sumKxy = sum(Kxy1);

    % Frobenius norms squared
    normKxxF2 = sum(Kxx(:).^2);
    normKyyF2 = sum(Kyy(:).^2);
    normKxyF2 = sum(Kxy(:).^2);

    % L2 norms squared of row sums
    normKxx1_2 = sum(Kxx1.^2);
    normKyy1_2 = sum(Kyy1.^2);
    normKxy1_2 = sum(Kxy1.^2);
    normKyx1_2 = sum(Kyx1.^2);

    % Mixed cross-covariance projections
    mix_XX_XY = Kxx1' * Kxy1;  % 1_n^T K_XX K_XY 1_m
    mix_YY_YX = Kyy1' * Kyx1;  % 1_m^T K_YY K_XY^T 1_n

    % Combinatorial denominators
    n2 = n*(n-1);    m2 = m*(m-1);
    n3 = n2*(n-2);   m3 = m2*(m-2);
    n4 = n3*(n-3);   m4 = m3*(m-3);

    % 4. First-Order Variance Components (nu_P and nu_Q)
    nu_P = (1/n3) * (normKxx1_2 - normKxxF2) ...
         - (2 / (m*n2)) * mix_XX_XY ...
         + (1 / (n*m2)) * (normKxy1_2 - normKxyF2) ...
         - (1/n4) * (sumKxx^2 - 4*normKxx1_2 + 2*normKxxF2) ...
         + (2 / (m*n3)) * (sumKxx * sumKxy - 2*mix_XX_XY) ...
         - (1 / (n2*m2)) * (sumKxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    nu_Q = (1/m3) * (normKyy1_2 - normKyyF2) ...
         - (2 / (n*m2)) * mix_YY_YX ...
         + (1 / (m*n2)) * (normKyx1_2 - normKxyF2) ...
         - (1/m4) * (sumKyy^2 - 4*normKyy1_2 + 2*normKyyF2) ...
         + (2 / (n*m3)) * (sumKyy * sumKxy - 2*mix_YY_YX) ...
         - (1 / (n2*m2)) * (sumKxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    V1 = (4/n) * nu_P + (4/m) * nu_Q;

    % 5. Second-Order Variance Components (tau_P, tau_Q, tau_PQ)
    tau_P = (1/n2) * normKxxF2 ...
          - (2/n3) * (normKxx1_2 - normKxxF2) ...
          + (1/n4) * (sumKxx^2 - 4*normKxx1_2 + 2*normKxxF2);

    tau_Q = (1/m2) * normKyyF2 ...
          - (2/m3) * (normKyy1_2 - normKyyF2) ...
          + (1/m4) * (sumKyy^2 - 4*normKyy1_2 + 2*normKyyF2);

    tau_PQ = (1/(n*m)) * normKxyF2 ...
           - (1/(n*m2)) * (normKxy1_2 - normKxyF2) ...
           - (1/(m*n2)) * (normKyx1_2 - normKxyF2) ...
           + (1/(n2*m2)) * (sumKxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    V2 = (2/n2) * tau_P + (2/m2) * tau_Q + (4/(n*m)) * tau_PQ;

    % 6. Result Packaging
    result.MMD2 = MMD2;
    result.sigma2_1 = V1;
    result.sigma2_2 = V2;
    result.sigma2 = V1 + V2;
    result.nuP = nu_P;
    result.nuQ = nu_Q;
end

%% Numerically Stable Kernel Function
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