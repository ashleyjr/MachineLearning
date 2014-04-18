function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=1;
	X=X_train(:,feat_in);						% Subset of features
	patterns=size(X,1);
	features=size(X,2);
	w_in=zeros(features,features);		% Weights from input to each hidden layer node
	w_out=zeros(features,1);			% Weights from hidden layer to output 

	for i=1:features
		w_out(i) = rand();
		for j=1:features
			w_in(i,) = rand()-0.5;
		end
	end


	w_in
	w_out

	param.w_in = w_in;
	param.w_out = w_out;
	feat_out = feat_in;
