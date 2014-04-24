function [best,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=1e-4;
	X=X_train(:,feat_in);						% Subset of features
	X_val=X_valid(:,feat_in);
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patternus
	features=size(X,2);
	hiddens=features*2;
	w_in=rand(hiddens,features)-0.5;		% Weights from input to each hidden layer node
	w_out=rand(1,hiddens);			% Weights from hidden layer to output 
	b_out = 0;


	low_valid = 1;	
	for i=1:200
		disp(i)


		% Forward
		for j=1:hiddens	
			Y_hidden(j,:) = tanh(X*w_in(j,:)');
		end
		Y = tanh(Y_hidden'*w_out')';



		% BackProp
		E = (Y_train' - Y);
		delta_in = eta.*E*X;
		for j=1:hiddens
			w_in(j,:) = w_in(j,:) + w_out(j)*delta_in;
		end
		delta_out = eta.*E*Y_hidden';
		w_out = w_out + delta_out;
		b_out = b_out - eta*sum(E);




		% performace
		x(i) = i;
		param.w_in = w_in(:,(1:(end-1)));
		param.b_in = w_in(:,end);
		param.w_out = w_out;
		param.b_out = b_out;
		[Y_resu_train, Y_conf_train] 	= predict( X_train, param,feat_in);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param,feat_in);
		err_train(i)					= balanced_errate(Y_resu_train, Y_train);
		err_valid(i)					= balanced_errate(Y_resu_valid, Y_valid);
		auc_train(i)					= auc(Y_resu_train.*Y_conf_train, Y_train);
    	auc_valid(i)					= auc(Y_resu_valid.*Y_conf_valid, Y_valid);
		if(err_valid(i) < low_valid)
			low_valid = err_valid(i);

			best.w_in = w_in(:,(1:(end-1)));
			best.b_in = w_in(:,end);
			best.w_out = w_out;
			best.b_out = b_out;
		end
	end



	figure;
	subplot(1,2,1);
	plot(x,err_train,x,err_valid);
	grid on;
	subplot(1,2,2);
	plot(x,auc_train,x,auc_valid);
	grid on;
	low_valid

	feat_out = feat_in;
