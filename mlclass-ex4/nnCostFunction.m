function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));  %25x401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));        %10x26

% Setup some useful variables

m = size(X,1);       %5000

classy = zeros(m, num_labels);        %5000x10
for i=1:num_labels
    classy(:,i) = y==i;
end

Thetax1 = Theta1(:,(2:end));          %25x400
Thetax2 = Theta2(:,(2:end));          %10x25
regterm = [Thetax1(:); Thetax2(:)]'*[Thetax1(:); Thetax2(:)]; %1x1
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];    %5000x400
z2 = a1*Theta1';                 %5000x25
a2 = sigmoid(z2);                %5000x25
a2 = [ones(m, 1) a2];  %5000x26
z3 = a2*Theta2';                 %5000x10
h = sigmoid(z3);                 %5000x10

for t=1:m
%     a1(t,:) = [1 X(t,:)];                %1x401
%     z2(t,:) = a1(t,:)*Theta1';           %1x401 * 401x25 = 1x25
%     a2(t,:) = [1 sigmoid(z2(t,:))];      %1x26
%     z3(t,:) = a2(t,:)*Theta2';           %1x26 * 26x10 = 1x10
%     h(t,:) = sigmoid(z3(t,:));           %1x10
    delta3(t,:) = h(t,:) - classy(t,:);  %1x10
    Theta2_grad = Theta2_grad + delta3(t,:)'*a2(t,:);      %10x26
    delta2(t,:) = (delta3(t,:)*Thetax2).*sigmoidGradient(z2(t,:)); %1x25
    Theta1_grad = Theta1_grad + delta2(t,:)'*a1(t,:);       %25x401
end

% -------------------------------------------------------------

% =========================================================================

% Calculate Cost and Unroll gradients

J = -((classy(:)'*(log(h(:))))+((1-classy(:))'*(log(1-h(:))))-(lambda*regterm/2))/m;
Theta1(:,1) = 0;
Theta2(:,1) = 0;
grad = ([Theta1_grad(:) ; Theta2_grad(:)] + lambda*[Theta1(:); Theta2(:)])/m;

end
