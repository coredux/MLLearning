function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    rightSum = zeros( length(theta) , 1 );
    for i = 1:m;
        xToi = X(i,:); %xToi 1 * n then xToi' n * 1
        % theta n * 1 then theta' 1 * n
        A = theta' * xToi';
        hypoToi = sum( A(:) );
        B = ( hypoToi - y(i) ) * xToi;
        rightSum = rightSum + B';
    end
    rightVal = rightSum * alpha / m;
    theta = theta - rightVal;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
