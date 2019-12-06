function [Ahat] = sketchyMNF(A, k)
    
    % Step 1
    % Initialize sketch matrices

    [m, n] = size(A);
    
    s = 2*k;

    X = sparse(k, n);
    Y = sparse(m, k);
    Z = sparse(s, s);
    Gamma = randn(k, m);
    Omega = randn(k, n);
    Phi = randn(s,m);
    Psi = randn(s,n);
    
    % Step 2
    % Stream data and store projections
        
    for i=1:n
       
       % One column each
       H = sparse(m, n);
       H(:,i) = A(:,i);

       X = X + Gamma * H;
       Y = Y + H * Omega';
       Z = Z + Phi * H * Psi';
    end
    
    % Step 3
    % Compute the core matrices
    
    % Only use X,Y,Z
    [Q,~]= qr(Y,0);
    [P,~]= qr(X',0);
    C1 = (Phi*Q);
    C2 = C1 \ Z;
    C3 = pinv(Psi*P);
    C = C2 * C3';
    
    AA = Q * C; % reconstruct row space

    % Step 4
    % Minimum Noise Fraction

    dX = zeros(size(AA));
    for i=1:(m-1)
        dX(i,:) = AA(i,:) - AA(i+1,:);
    end

    % noise covariance decomposition 
    [U1,S1,~] = svd(dX'*dX);
    D1 = diag(S1);
    diagS1 = 1./sqrt(D1); %store vector form since it is a diagonal matrix

    % Whiten the original data
    wX = AA*U1*diag(diagS1); %diagS1 is much cheaper than the inversion

    [U2,S2,~] = eig(wX'*wX);
    [D2,iy] = sort(diag(S2),'descend');
    U2 = U2(:,iy);

    % matrix S2=SNR+1, which provides a good esimate of r
    S2_diag = (D2)-1;
    i = (sum(S2_diag>5.0)); %according to Rose's criteria

    r = i; %number of top r components to retain
    disp(['K: ' num2str(r)]);
    U2 = U2(:,1:r);

    % compute the MNF basis vectors
    Phi_hat = U1*diag(diagS1)*U2;
    Phi_tilde = U1*diag(sqrt(D1))*U2;

    % MNF and inverse MNF
    Xhat = AA * Phi_hat * Phi_tilde';
    Ahat = Xhat * P'; % reconstruct column space

end
