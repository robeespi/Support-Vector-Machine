%Load training data
load('train.mat');
%Handling variables for convenience
y(y==0) = -1;
labelstrain=y';
featurestrain = X;
%Sparsing the data, according to the format required by library
fstrain = sparse(featurestrain);
%Writing the data into a libsvm file
libsvmwrite('svmrobtrain', labelstrain, fstrain);
%Defining variables
[ytrain,xtrain]=libsvmread('svmrobtrain');
trainlabel=ytrain;
traindata=xtrain;
%Defining the train model
model = svmtrain(trainlabel, traindata);

%Loading test data
load('test.mat');
%Handling variables for convenience
y(y==0) = -1;
labelstest=y';
featurestest = X;
%Sparsing the data, according to the format required by library
fstest = sparse(featurestest);
%Writing the data into a libsvm file
libsvmwrite('svmrobtest', labelstest, fstest);
%Defining variables
[ytest,xtest]=libsvmread('svmrobtest');
testlabel=ytest;
testdata=xtest;
%Storing the test output from libsvm
[predicted_label_test] = svmpredict(testlabel, testdata, model);



