function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
rightSum = 0;

for i=1:m;
    xToi = X(i,:); %xToi 1 * n then xToi' n * 1
    A = theta' * xToi';
    hypoToi = sum( A(:) ); % theta n * 1 then theta' 1 * n
    rightSum = rightSum + ( y(i) - hypoToi )^2;
end
J = rightSum / ( 2 * m );


% =========================================================================

end
