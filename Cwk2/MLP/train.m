function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	X=X_train(:,feat_in);						% Subset of features
	patterns=size(X,1);
	features=size(X,2);
	param.w_in=rand(features,features)-0.5;		% Weights from input to each hidden layer node
	param.w_out=rand(1,features,1)-0.5;			% Weights from hidden layer to output 
	param.b_in = 0;
	param.b_out = 0;

	minErr = 1;

	for i=1:5000	
		disp(i)
		w_in_move=10*(rand(features,features)-0.5);		% Weights from input to each hidden layer node
		w_out_move=10*(rand(1,features)-0.5);			% Weights from hidden layer to output 
		b_in_move = rand()-0.5;
		b_out_move = rand()-0.5;

		param.w_in = param.w_in + w_in_move;
		param.w_out = param.w_out + w_out_move;
		param.b_in = param.b_in + b_in_move;
		param.b_out = param.b_out + b_out_move;

		
		[resu, conf] = predict(X_train,param,feat_in);
		err(i)=balanced_errate(resu, Y_train);
		
		if(err(i)<minErr)
			minErr = err(i);
		else
			param.w_in = param.w_in - w_in_move;
			param.w_out = param.w_out - w_out_move;
			param.b_in = param.b_in - b_in_move;
			param.b_out = param.b_out - b_out_move;
		end
	end

	figure;
	plot(err)
	
	feat_out = feat_in;
