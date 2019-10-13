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
K=Xt*Xt'; %nxn
%Choosing a feasible c
Ct=0.001;
%Using cvx for solving
cvx_begin %dual
    variables alphatrain2(n) %you don't have anything with size d
    maximize( sum(alphatrain2) -  0.5*quad_form(Yt.*alphatrain2,K))
    subject to
       alphatrain2>0
       alphatrain2<Ct
       sum(alphatrain2.*Yt)==0
cvx_end

%Calculing variables
wtrain2=Xt'*(alphatrain2.*Yt);
epsilon=0.0001;
svii = find( alphatrain2 > epsilon & alphatrain2 < (Ct - epsilon));
btrain2 =  (1/length(svii))*sum(Yt(svii) - K(svii,:)*alphatrain2.*Yt(svii));

%Loading test data
load('test.mat');
%Transpose Y for convenience
Ytest=y';
%Replace 0 by -1
Ytest(Ytest==0) = -1;
%Calculating error and acccuracy
Xtest=X;
Ktest=Xtest*Xt';
predictedY2=sign(Ktest*(alphatrain2.*Yt)+btrain2);
errororig=sum(Ytest~=predictedY2)/size(Ytest,1);
accuracy = (1-errororig)*100;



