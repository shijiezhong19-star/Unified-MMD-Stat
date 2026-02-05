function result = MMD_propose(X, Y, kernel_type, param)

% MMD_PROPOSE - Unified finite-sample unbiased MMD^2 and variance estimator.
%
% This function implements the proposed euMMD algorithm, which provides a
% unified characterization of MMD variance across null and alternative 
% hypotheses, supporting both balanced and imbalanced sample sizes.
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
%   result.sigma2   - Total unified variance (sigma2_t1 + sigma2_t2)
%   result.sigma2_1 - First-order component (from Hoeffding decomposition)
%   result.sigma2_2 - Second-order component (within-sample fluctuation)
%   result.rho1     - Ratio of first-order variance to total variance

    if size(X,2) ~= size(Y,2)
        error('X and Y must have the same feature dimension.');
    end

    n = size(X,1);
    m = size(Y,1);

    % 1. Kernel Matrix Computation
    Kxx = kernel_func(X, X, kernel_type, param);
    Kyy = kernel_func(Y, Y, kernel_type, param);
    Kxy = kernel_func(X, Y, kernel_type, param);

    % Remove diagonal elements to satisfy the i ~= j condition for U-statistics
    Kxx(1:n+1:end) = 0;
    Kyy(1:m+1:end) = 0;

    % 2. Unbiased MMD^2 Estimation
    % Based on the two-sample U-statistic with kernel h
    MMD2 = sum(Kxx(:))/(n*(n-1)) ...
        + sum(Kyy(:))/(m*(m-1)) ...
        - 2*sum(Kxy(:))/(n*m);

    % 3. First-Order Variance Component (sigma2_t1)
    % Derived from the influence functions in Hoeffding decomposition
    one_m = ones(n,1);
    one_n = ones(m,1);
    Kxx1 = Kxx * one_m;
    Kyy1 = Kyy * one_n;
    Kxy1 = Kxy * one_n;
    Kyx1 = Kxy' * one_m;

    % Influence function components U and V
    U = Kxx1/(n-1) - Kxy1/m;
    V = Kyy1/(m-1) - Kyx1/n;
    varU = var(U); 
    varV = var(V);

    % sigma2_t1 accounts for the leading O(n^-1 + m^-1) term
    sigma2_t1 = 4*(n-2)/(n*(n-1)) * varU + 4*(m-2)/(m*(m-1)) * varV;

    % 4. Second-Order Variance Component (sigma2_t2)
    % Precompute quadratic forms and Frobenius norms for exact estimation
    normKxxF2 = sum(sum(Kxx.^2));    
    normKxx1_2 = sum(Kxx1.^2);     
    sumKxx = sum(Kxx1);            

    normKyyF2 = sum(sum(Kyy.^2));  
    normKyy1_2 = sum(Kyy1.^2);      
    sumKyy = sum(Kyy1);        

    normKxyF2 = sum(sum(Kxy.^2));   
    normKxy1_2 = sum(Kxy1.^2);     
    normKyx1_2 = sum(Kyx1.^2);    
    sumKxy = sum(Kxy1);

    % Combinatorial denominators for unbiasedness
    n2 = n*(n-1);
    n3 = n2*(n-2);
    n4 = n3*(n-3);

    m2 = m*(m-1);
    m3 = m2*(m-2);
    m4 = m3*(m-3);

    % Exact second-order expectations (Ug-statistics)
    % EgA2, EgB2, EgC2 correspond to the within-sample and cross-sample variance cores
    EgA2 = (1/n2) * normKxxF2 ...
        - (2/n3) * (normKxx1_2 - normKxxF2) ...
        + (1/n4) * (sumKxx^2 - 4*normKxx1_2 + 2*normKxxF2);

    EgB2 = (1/m2) * normKyyF2 ...
        - (2/m3) * (normKyy1_2 - normKyyF2) ...
        + (1/m4) * (sumKyy^2 - 4*normKyy1_2 + 2*normKyyF2);

    EgC2 = (1/(n*m)) * normKxyF2 ...
        - (1/(n*m2)) * (normKyx1_2 - normKxyF2) ...
        - (1/(m*n2)) * (normKxy1_2 - normKxyF2) ...
        + (1/(n2*m2)) * (sumKxy^2 - normKxy1_2 - normKyx1_2 + normKxyF2);

    % sigma2_t2 accounts for the higher-order O(n^-2 + m^-2) term
    sigma2_t2 = 2/(n*(n-1))*EgA2 + 2/(m*(m-1))*EgB2 + 4/(n*m) * EgC2;

    % 5. Result Packaging

    result.MMD2 = MMD2;
    result.sigma2 = sigma2_t1 + sigma2_t2;
    result.sigma2_1 = sigma2_t1;
    result.sigma2_2 = sigma2_t2;
    result.rho1 = sigma2_t1/(sigma2_t1+ sigma2_t2);
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