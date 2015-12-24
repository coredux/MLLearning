function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sumJVal = 0;

for i = 1:m;
    sumJVal = sumJVal + ( -y(i) * log( sigmoid( X( i , : ) * theta ) ) - ( 1 - y(i) ) * log( 1 - sigmoid( X( i , : ) * theta ) ) );
end

J = sumJVal / m;

n = size( theta );
sumRight = 0;
for i = 2:n;
    sumRight = sumRight + theta( i )^2;
end

J = J + ( lambda * sumRight ) / ( 2 * m );

for i = 1:size(theta);
    jVal = 0;
    for j = 1:m;
        jVal = jVal + ( sigmoid( X( j , : ) * theta ) - y( j ) ) * X( j , i );
    end
    jVal = jVal / m;
    if i > 1;
        jVal = jVal + lambda * theta( i ) / m;
    end
    grad( i ) = jVal;
end





% =============================================================

end
