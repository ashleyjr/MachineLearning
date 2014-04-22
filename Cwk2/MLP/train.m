function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=5e-8;
	X=X_train(:,feat_in);						% Subset of features
	patterns=size(X,1);
	features=size(X,2);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	w_in=rand(features+1,features+1)-0.5;		% Weights from input to each hidden layer node
	w_out=rand(1,features+1,1)-0.5;			% Weights from hidden layer to output 
	param.b_out = 0;
	param.b_in = 0;


	for i=1:10000
		disp(i)

		for j=1:patterns
			Y(j) = w_out*(tanh(w_in*X(j,:)'));
		end
		Y = unitVec(Y);
		err(i)=balanced_errate(Y', Y_train);

		deltaW = eta*((Y_train - Y')'*X);	
		w_out_new = w_out + deltaW;
		for i=1:features
			deltaW = eta*w_out(i)*((Y_train - Y')'*X);
			w_in(i,:) = w_in(i,:) + deltaW;
		end
		w_out = w_out_new;
	
	end
	for j=1:patterns
		Y(j) = w_out*(tanh(w_in*X(j,:)'));
	end
	Y = unitVec(Y);
	err(i)=balanced_errate(Y', Y_train);

	w_out
	w_in
	param.w_out = w_out(1,1:(end-1));
	param.b_out = w_out(end);
	param.w_in = w_in(1:(end-1),1:(end-1));
	param.b_in = w_in(end,1:(end-1))';
	figure;
	plot(err)
	
	feat_out = feat_in;
