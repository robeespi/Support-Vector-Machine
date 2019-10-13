%Loading the data
load('train.mat');
%Handling variables for convenience
y=y';
y(y==0) = -1;
X=X;
Y=y;
K=X*X'; % Dot-product 
N = size(X,1);
    %Using cvx
    cvx_begin %dual problem
        cvx_precision best
        variable alphad(N);
        minimize (0.5.*quad_form(Y.*alphad,K) - ones(N,1)'*(alphad));
        subject to
            alphad >= 0;
            Y'*(alphad) == 0;
    cvx_end

   
    
