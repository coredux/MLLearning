function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
CVec = [ 0.01 , 0.03 , 0.1 , 0.3 , 1 , 3 , 10 , 30 ];
sigmaVec = [ 0.01 , 0.03 , 0.1 , 0.3 , 1 , 3 , 10 , 30 ];
minError = 0;
nowError = 0;
firstTime = 1;

for i = 1:length(CVec);
    for j = 1:length(sigmaVec);
        
        nowC = CVec(i);
        nowSigma = sigmaVec(j);
        
        model = svmTrain(X, y, nowC, @(x1, x2) gaussianKernel(x1, x2, nowSigma));
        predictions = svmPredict( model , Xval );
        nowError = mean(double(predictions ~= yval));
        
        if firstTime == 1;
            minError = nowError;
            firstTime = 0;
        end
        
        if nowError < minError;
            C = nowC;
            sigma = nowSigma;
            minError = nowError;
        end
    end
end







% =========================================================================

end
