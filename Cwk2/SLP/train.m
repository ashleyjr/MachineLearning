function [param, idx_out]=SLP_train(X_train, Y_train, X_valid, Y_valid, idx_in)
	eta=0.0001;
	X=X_train(:,idx_in);			% Subset of features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);			% Offset on all input patterns
	features=size(X,2);
	W=zeros(features,1);			% Init weights 


	for t=1:200
		x(t) = t;
		Y=unitVec(X*W)';	
		
		
		deltaW = eta*((Y_train - Y)'*X)';
		W = W + deltaW;

		param.W = W(1:(end-1));
		param.b = W(end);

		[Y_resu_train, Y_conf_train] 	= predict( X_train, param, idx_in, X_train, Y_train);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param, idx_in, X_train, Y_train);

		% Blance error for train and validiate
		err_bal_train(t)				= balanced_errate(Y_resu_train, Y_train);
    	err_bal_valid(t)				= balanced_errate(Y_resu_valid, Y_valid);
		
		% AUC error for train and validiate
		err_auc_train(t)				= auc(Y_resu_train.*Y_conf_train, Y_train);
    	err_auc_valid(t)				= auc(Y_resu_valid.*Y_conf_valid, Y_valid);
	end

	figure(10)
	subplot(1,2,1)
	plot(x,err_bal_train,x,err_bal_valid)
	subplot(1,2,2)
	plot(x,err_auc_train,x,err_auc_valid)


	for t=1:1000
		x(t) = t;
		[Y_resu_train, Y_conf_train] 	= predict( X_train, param, idx_in, X_train, Y_train);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param, idx_in, X_train, Y_train);

		% Blance error for train and validiate
		err_bal_train(t)				= balanced_errate(Y_resu_train, Y_train);
    	err_bal_valid(t)				= balanced_errate(Y_resu_valid, Y_valid);
		
		% AUC error for train and validiate
		err_auc_train(t)				= auc(Y_resu_train.*Y_conf_train, Y_train);
    	err_auc_valid(t)				= auc(Y_resu_valid.*Y_conf_valid, Y_valid);


		% mutate
		pick=ceil(rand(1)*size(param.W));
		move=rand(1)-0.5;

		param_test = param;
		param_test.W(pick)=param_test.W(pick)+move;	
	
		[Y_resu_train, Y_conf_train] 	= predict( X_train, param_test, idx_in, X_train, Y_train);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param_test, idx_in, X_train, Y_train);

		err_bal_valid_test				= balanced_errate(Y_resu_valid, Y_valid);
		err_auc_valid_test				= auc(Y_resu_valid.*Y_conf_valid, Y_valid);

		if((err_bal_valid_test < err_bal_valid(t)) || (err_auc_valid_test > err_auc_valid(t))  )
			param = param_test;
		end


		

	end

	figure(11)
	subplot(1,2,1)
	plot(x,err_bal_train,x,err_bal_valid)
	subplot(1,2,2)
	plot(x,err_auc_train,x,err_auc_valid)


	idx_out=idx_in;
