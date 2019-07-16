% Efficient Sketchy SVD implementation of Joel Tropp's paper
% return low rank breakdown of extremely large matrix

% Input
% A - Input Matrix
% r - Estimated Rank
% k - Overestimated Rank

% Output
% U - Approximated Rank-r Left Singular Vectors
% S - Approximated Rank-r Singular Values
% V - Approximated Rank-r Right Singular Values

%%
function [U, Sigma, V] = sketchy_svd(A, r, k)
    
    % Step 1
    [m, n] = size(A);
    s = 2 * k;
    
    % Assignment
    X = sparse(k, n);
    Y = sparse(m, k);
    Z = sparse(s, s);
    Gamma = randn(k, m);
    Omega = randn(k, n);
    Phi = randn(s,m);
    Psi = randn(s,n);
    
    % Step 2
    list = randperm(n); % randomize order of streaming
    
    for i=1:n
       
       % One column each
       H = sparse(m, n);
       H(:,list(i)) = A(:,list(i)); % read random columns

       X = X + Gamma * H; % row projection 
       Y = Y + H * Omega'; % column projection
       Z = Z + Phi * H * Psi'; % core (row-column) projection
    end
    
    % Step 3
    
    % Only use X,Y,Z
    [Q,~]= qr(Y,0); % economic QR of column space
    [P,~]= qr(X',0); % economic QR of row space
    C1 = (Phi*Q);
    C2 = C1 \ Z;
    C3 = pinv(Psi*P);
    C = C2 * C3'; % core sketch
    
    % Step 4
    
    % SVD of core sketch
    % One can use a fancier version of SVD too
    [Uhat,Sigma,Vhat] = svd(C);
    
    % Project in original space
    U = Q * Uhat;
    V = P * Vhat;
    
end

