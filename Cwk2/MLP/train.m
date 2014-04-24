function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=1e-5;
	X=X_train(:,feat_in);						% Subset of features
	X_val=X_valid(:,feat_in);
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patternus
	X_val=horzcat(X_val,ones_row);
	features=size(X,2);
	hiddens=features;
	w_in=rand(hiddens,features)-0.5;		% Weights from input to each hidden layer node
	w_out=rand(1,hiddens);			% Weights from hidden layer to output 
	param.b_out = 0;
	param.b_in = 0;


	
	for i=1:500
		disp(i)


		% Forward
		for j=1:hiddens	
			Y_hidden(j,:) = tanh(X*w_in(j,:)');
		end
		Y = tanh(Y_hidden'*w_out')';



		% BackProp
		E = (Y_train' - Y);
		delta_out = eta.*E*X;
		for j=1:hiddens
			w_in(j,:) = w_in(j,:) + w_out(j)*delta_out;
		end
		w_out = w_out + delta_out;




		% performace


		for j=1:hiddens	
			Y_hidden(j,:) = tanh(X_val*w_in(j,:)');
		end
		Y_val = unitVec(Y_hidden'*w_out');
		% Forward
		for j=1:hiddens	
			Y_hidden(j,:) = tanh(X*w_in(j,:)');
		end
		Y = unitVec(Y_hidden'*w_out');
		err_train(i)=balanced_errate(Y', Y_train);
		err_valid(i)=balanced_errate(Y_val', Y_valid);

		x(i) = i;
	end



	figure;
	plot(x,err_train,x,err_valid);

	param.w_in = w_in(:,(1:(end-1)));
	param.b_in = w_in(:,end);
	param.w_out = w_out;
	param.b_out = 0;

	feat_out = feat_in;
