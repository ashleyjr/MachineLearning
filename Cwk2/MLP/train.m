function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=1e-6;
	X=X_train(:,feat_in);						% Subset of features
	x_val=X_valid(:,feat_in);
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patternus
	x_val=horzcat(x_val,ones_row);					% Offset on all input patterns
	features=size(X,2);
	w_in=ones(features,features)-0.5;		% Weights from input to each hidden layer node
	w_out=ones(1,features,1)-0.5;			% Weights from hidden layer to output 
	param.b_out = 0;
	param.b_in = 0;


	for j=1:patterns
		Y(j) = w_out*(tanh(w_in*X(j,:)'));
	end
	Y = unitVec(Y);
	
	for i=1:600
		disp(i)

		
		% AJR - Use tanh to feedback error?, don;t use _|- 



		% Output layer
		delta_out = (Y_train' - Y);	
		w_new_out = w_out + eta*delta_out*X;


		% Hidden layer
		for k=1:features
			delta_in = w_out(k)*delta_out;
			w_in(k,:) = w_in(k,:) + eta*delta_in'*X;
		end

		w_out = w_new_out;



		% performace
		param.w_out = w_out(1,1:(end-1));
		param.b_out = w_out(end);
		param.w_in = w_in(1:(end-1),1:(end-1));
		param.b_in = w_in(end,1:(end-1))';


    	%[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param,	feat_in	);
		%[Y_resu_train, Y_conf_train] 	= predict( X_train, param,	feat_in	);
		
		for j=1:patterns
			Y(j) = w_out*(tanh(w_in*X(j,:)'));
		end
		Y = unitVec(Y);

		for j=1:patterns
			y_val(j) = w_out*(tanh(w_in*x_val(j,:)'));
		end
		y_val = unitVec(y_val);


		err_valid(i)=balanced_errate(y_val', Y_valid);
		err_train(i)=balanced_errate(Y', Y_train);
		x(i) = i;
	end



	figure;
	plot(x,err_train,x,err_valid)

	
	feat_out = feat_in;
