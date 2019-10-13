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

%Using cvx for solving
cvx_begin  %Primal
    variables wtrain(d) btrain
    dual variable alphatrain
    minimize( 0.5*wtrain'*wtrain ) 
    subject to
        Yt.*(Xt*wtrain+btrain) + btrain > 1;   
cvx_end
