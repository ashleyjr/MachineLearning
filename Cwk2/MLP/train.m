function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta_hidden=1e-7;
	eta_output=eta_hidden*10;
	X=X_train(:,feat_in);						% Subset of features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	features=size(X,2);
	w_in=ones(features,features)-0.5;		% Weights from input to each hidden layer node
	w_out=ones(1,features,1)-0.5;			% Weights from hidden layer to output 
	param.b_out = 0;
	param.b_in = 0;


	for i=1:2000
		disp(i)


		% Hidden layer
		for j=1:patterns
			Y(j) = w_out*(tanh(w_in*X(j,:)'));
		end
		Y = unitVec(Y);
		for k=1:features
			deltaW = eta_hidden*w_out(k)*((Y_train - Y')'*X);
			w_in(k,:) = w_in(k,:) + deltaW;
		end


		% Output layer
		for j=1:patterns
			Y(j) = w_out*(tanh(w_in*X(j,:)'));
		end
		Y = unitVec(Y);
		deltaW = eta_output*((Y_train - Y')'*X);	
		w_out = w_out + deltaW;


		% performace
		err_train(i)=balanced_errate(Y', Y_train);

		param.w_out = w_out(1,1:(end-1));
		param.b_out = w_out(end);
		param.w_in = w_in(1:(end-1),1:(end-1));
		param.b_in = w_in(end,1:(end-1))';

    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param,	feat_in	);

		err_valid(i)=balanced_errate(Y_resu_valid, Y_valid);
		x(i) = i;
	end



	figure;
	plot(x,err_train,x,err_valid)

	
	feat_out = feat_in;
