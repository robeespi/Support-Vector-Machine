%Load the data
load('train.mat');
%Transpose y for convenience
y=y';
%Replace 0 by -1
y(y==0) = -1;
%Reasigning variables
X=X;
Y=y;
[n,d] = size (X);
Yt=Y; 
Xt=X; 
%Choosing a feasible c
Ct=0.001;
%Using cvx for solving
cvx_begin  %Primal
    variables wtrain(d) e(n) btrain
    dual variable alphatrain
    minimize( 0.5*wtrain'*wtrain + Ct*sum(e)) %norm(w) almost works except it takes an extra sqrt
    subject to
        Yt.*(Xt*wtrain+btrain)-1 +e >0;
        e>0; %slack
cvx_end

