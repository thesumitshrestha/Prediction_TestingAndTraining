Theta1 = load('Theta1.txt');
Theta1 = Theta1.Theta1;
Theta2 = load('Theta2.txt');
Theta2 = Theta2.Theta2;
X = dlmread('Test.csv');
y = dlmread('TestResult.csv');

#X = [1.02 0.18];
#y = [0	1 0];

a1 = [ones(size(X(:, 1)), 1) X]';
a2 = sigmoid(Theta1 * a1)';
a2 = [ones(size(a2(:, 1)), 1) a2]';
a3 = sigmoid(Theta2 * a2)';
[dummy, p1] = max(a3, [], 2);
[dummy, p2] = max(y, [], 2);
mean(double(p1 == p2) * 100)
