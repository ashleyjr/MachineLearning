function [best,feat_out]=trainMLP(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta_in=1e-4;
	eta_out=1e-4;
	X=X_train(:,feat_in);						% Subset of features
	X_val=X_valid(:,feat_in);
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patternus
	features=size(X,2)
	hiddens=1000;
	w_in=rand(hiddens,features)-0.5;		% Weights from input to each hidden layer node
	w_out=rand(1,hiddens)-0.5;			% Weights from hidden layer to output 
	b_out = 0;
	low_valid = 100;
	for i=1:80
		disp(i)
		% Forward
		for j=1:hiddens	
			Y_hidden(j,:) = tanh(X*w_in(j,:)');
		end
		Y = tanh(Y_hidden'*w_out')';
		% BackProp
		E = (Y_train' - Y);
		delta_in = eta_in.*E*X;
		for j=1:hiddens
			w_in(j,:) = w_in(j,:) + w_out(j)*delta_in;
		end
		delta_out = eta_out.*E*Y_hidden';
		w_out = w_out + delta_out;
		b_out = b_out - eta_out*sum(E);
		% performace
		x(i) = i;
		param.w_in = w_in(:,(1:(end-1)));
		param.b_in = w_in(:,end);
		param.w_out = w_out;
		param.b_out = b_out;
		[Y_resu_train, Y_conf_train] 	= predictMLP( X_train, param,feat_in);
    	[Y_resu_valid, Y_conf_valid] 	= predictMLP( X_valid, param,feat_in);
		err_train(i)					= balanced_errate(Y_resu_train, Y_train)*100;
		err_valid(i)					= balanced_errate(Y_resu_valid, Y_valid)*100;
		auc_train(i)					= auc(Y_resu_train.*Y_conf_train, Y_train)*100;
    	auc_valid(i)					= auc(Y_resu_valid.*Y_conf_valid, Y_valid)*100;
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
	xlabel('Iterations')
	ylabel('BER(%)')
	grid on;
	subplot(1,2,2);
	plot(x,auc_train,x,auc_valid);
	xlabel('Iterations')
	ylabel('AUC(%)')
	grid on;
	feat_out = feat_in;
