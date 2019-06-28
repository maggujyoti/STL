function [T, Z] = lrTransformLearning (X, labels, numOfAtoms, mu, lambda, eps)

% solves ||TX - Z||_Fro - mu*logdet(T) + eps*mu||T||_Fro + lambda||Z||_NN

% Inputs
% X          - Training Data
% labels     - Class labels
% numOfAtoms - dimensionaity after Transform
% mu         - regularizer for Tranform
% lambda     - regularizer for coefficient
% eps        - regularizer for Transform
% type       - 'soft' or 'hard' update: default is 'soft'
% Output
% T          - learnt Transform
% Z          - learnt sparse coefficients

if nargin < 6
    eps = 1;
end
if nargin < 5
    lambda = 0.1;
end
if nargin < 4
    mu = 0.1;
end

maxIter = 10;
type = 'soft'; % default 'soft'

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));

invL = (X*X' + mu*eps*eye(size(X,1)))^(-0.5);

for i = 1:maxIter
    
    % update Coefficients Z
    Z = T*X;
    for k = min(labels):max(labels)
        idx = find(labels == k);
        [U1,S1,V1] = svd(Z(:,idx),'econ');
        switch type
            case 'soft'
                S1 = diag(max(0,diag(S1)-lambda)); % soft thresholding
            case 'hard'
                S1 = diag((diag(S1) >= lambda).*diag(S1)); % hard thresholding
        end
        Z(:,idx) = U1*S1*V1';
        clear idx
    end
    
    

    % update Transform T
    [U,S,V] = svd(invL*X*Z');
    D = [diag(diag(S) + (diag(S).^2 + 2*mu).^0.5) zeros(numOfAtoms, size(X,1)-numOfAtoms)];
    T = 0.5*V*D*U'*invL;
    
end