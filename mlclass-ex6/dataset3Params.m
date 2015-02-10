function [C, sigma] = dataset3Params(X, y, Xval, yval, x1, x2)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Ctest = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmatest = [0.01 0.03 0.1 0.3 1 3 10 30];
errors = zeros(size(Ctest,2), size(sigmatest,2));

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

for i=1:size(Ctest,2)
    for j=1:size(sigmatest,2)
        model= svmTrain(X, y, Ctest(i), @(x1, x2) gaussianKernel(x1, x2, sigmatest(j))); 
        predictions = svmPredict(model, Xval);
        errors(i,j) = mean(double(predictions ~= yval));
        clear model;
        clear predictions;
    end
end

[dummy,ind] = min(errors(:));
[i,j] = ind2sub([size(errors,1) size(errors,2)],ind);
C = Ctest(i);
sigma = sigmatest(j);

% =========================================================================

end
