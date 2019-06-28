function [T, Z] = gsTransformLearning (X, labels, numOfAtoms, mu, lambda, eps)

% solves ||TX - Z||_Fro - mu*logdet(T) + eps*mu||T||_Fro + lambda||Z||_2,1

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
type = 'hard'; % default 'soft'

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));

invL = (X*X' + mu*eps*eye(size(X,1)))^(-0.5);

for i = 1:maxIter
    
    % update Coefficients Z
    Z = T*X;
    for k = min(labels):max(labels)
        idx = find(labels == k);
        for i = 1:size(Z,1)
                Thres(i,:) = (1/norm(Z(i,idx))).*abs(Z(i,idx)); % compute threshold                  
        end
        switch type
            case 'soft'
                Z(:,idx) = sign(Z(:,idx)).*max(0,abs(Z(:,idx))-lambda*Thres); % soft thresholding
            case 'hard'
                Z(:,idx) = abs(Z(:,idx) >= lambda*Thres).*Z(:,idx); % hard thresholding
        end
        clear Thres idx
    end
    
    

    % update Transform T
    [U,S,V] = svd(invL*X*Z');
    D = [diag(diag(S) + (diag(S).^2 + 2*mu).^0.5) zeros(numOfAtoms, size(X,1)-numOfAtoms)];
    T = 0.5*V*D*U'*invL;
    
end