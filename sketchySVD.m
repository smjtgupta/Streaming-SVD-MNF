function [Ahat] = sketchySVD(A,r)

    % Step 1
    % Initialize sketch matrices

    [m, n] = size(A);
    k = 2*r;
    s = 2*k;

    X = zeros(k, n);
    Y = zeros(m, k);
    Z = zeros(s, s);
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
    
    % Step 4
    % Compute rank-r approximation
    
    epsilon = 0.01;
    spectral_gap = floor(log(n/epsilon));
    [U,Shat,V] = BlockLanczos(C,r,spectral_gap);

    Uhat = Q*U;
    Vhat = P*V;
    Ahat = Uhat * Shat * Vhat'; % return low rank sketch

end

