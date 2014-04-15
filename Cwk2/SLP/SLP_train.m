function [param, idx_out]=SLP_train(X_train, Y_train, idx_in)
	eta=0.001
	X=X_train(:,idx_in);			% Subset of features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);			% Offset on all input patterns
	features=size(X,2);
	W=zeros(features,1);			% Init weights 	
	minErrate=1;
	for t=1:100
		Y=unitVec(X*W)';						% perceptron
		errate(t) = balanced_errate(Y, Y_train);	
		deltaW = eta*((Y_train - Y)'*X)';
		W = W + deltaW;
	end

	figure(2)
	plot(errate)
	minErrate
	param.W = W(1:(end-1));
	param.b = W(end);
	idx_out=idx_in;
