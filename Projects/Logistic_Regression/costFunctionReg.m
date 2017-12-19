function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
k=size(X);
n=k(:,2);
reg=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=2:n
                                reg=reg + theta(i,1)*theta(i,1);
endfor 

reg=reg .* (lambda/(2*m) );
X_sig_reg=sigmoid(X*theta);

J = -(    sum((log(X_sig_reg) .* y )) +       sum((log(1-X_sig_reg).* (1-y)))    ) ./ m + reg;

grad=sum( (X_sig_reg-y) .* X ) ./ m ;


j=0;
for j=2:n
grad(1,j)=grad(1,j) + ( (lambda/m)*theta(j,1) );
endfor

% =============================================================

end




%X_sig=sigmoid(X*theta);
%X_sig=1 ./ (1+ exp(- (X*theta) ) );
%J = -(    sum((log(X_sig) .* y )) +       sum((log(1-X_sig).* (1-y)))    ) ./ m;

%grad=transpose(sum(transpose(X_sig-y .* X ))) ./ m;
%grad=sum( (X_sig-y) .* X ) ./ m;

