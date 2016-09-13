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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(size(X(1, :))); X];
a2 = sigmoid(Theta1 * a1)';
a2 = [ones(size(a2(:, 1)), 1) a2]';
a3 = sigmoid(Theta2 * a2);

J = sum(sum(-y .* log(a3) - (1 - y) .* log(1 - a3))) / m;
J += lambda * (sum(sum(Theta1(:, [2:end]) .** 2)) + sum(sum(Theta2(:, [2:end]) .** 2))) / (2 * m);

% -------------------------------------------------------------
%Backpropagation

delta3 = a3 - y;
delta2 = (Theta2(:, [2:end])' * delta3) .* sigmoidGradient(Theta1 * a1);

Theta1_grad = (delta2 * a1') / m;
Theta2_grad = (delta3 * a2') / m;
Theta1_grad(:, [2:end]) += (lambda * Theta1(:, [2:end])) / m;
Theta2_grad(:, [2:end]) += (lambda * Theta2(:, [2:end])) / m;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
