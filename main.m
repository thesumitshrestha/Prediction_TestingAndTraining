%% Initialization
clear ; close all;

%% Setup the parameters you will use for this exercise

input_layer_size  = 2;
hidden_layer_size = 10;
num_labels = 3;

X = dlmread('Train.csv', ',');
X = X';
y = dlmread('TrainResult.csv', ',');
y = y';
m = size(X, 1);
fprintf('\nInitializing Neural Network Parameters ...\n')
%epsilon_init = 0.12  *(2*epsilon_init)-epsilon_init ;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 100000);

lambda = 0.00001;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save 'Theta1' Theta1
save 'Theta2' Theta2
