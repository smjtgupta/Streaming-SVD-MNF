function [Xhat] = MNF(X)

    % noise estimation
    [m, n] = size(X);
    dX = zeros(m,n);
    for i=1:(m-1)
        dX(i,:) = X(i,:) - X(i+1,:);
    end
    
%     figure;
%     imagesc(cov(dX));colorbar;
    
    % noise covariance decomposition 
    [U1,S1,~] = svd(dX'*dX);
    D1 = diag(S1);
%     [D1,ix] = sort(diag(S1),'descend');
%     U1 = U1(:,ix);
    diagS1 = 1./sqrt(D1); %store vector form since it is a diagonal matrix

    % Whiten the original data
    wX = X*U1*diag(diagS1); %diagS1 is much cheaper than the inversion

    % Compute the eigenvector expansion of the covariance of wX
    %[U2,S2,V2] = svd(wX'*wX);
    % since the covariance matrix is square symmetric, 
    % eigen decomposition gives faster results
    [U2,S2,~] = eig(wX'*wX);
    [D2,iy] = sort(diag(S2),'descend');
    U2 = U2(:,iy);

    % matrix S2=SNR+1, which provides a good esimate of K
    S2_diag = (D2)-1;
    i = (sum(S2_diag>5.0)); %according to Rose's criteria

    K = i; %number of top K components to retain
    U2 = U2(:,1:K);

    % compute the MNF basis vectors
    Phi_hat = U1*diag(diagS1)*U2;
    Phi_tilde = U1*diag(sqrt(D1))*U2;

    % transform data into MNF space
    mnfX = X*Phi_hat;

    % inverse MNF
    % tranform data back into signal space
    Xhat = mnfX * Phi_tilde';
    
    vars = whos('X','dX','U1','Xhat');
    sum_vars = 0;
    for j= 1:numel(vars)
        sum_vars = sum_vars + vars(j).bytes / 1024^2;
    end
    disp(sum_vars)

end

