function [param,idx_feat]=train(X_train, Y_train, X_valid, Y_valid, feat)
	eta=1e-4;
	X=X_train(:,feat);						% All features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	features=size(X,2);
	W=rand(features,1)-0.5;					% Init weights to zero 
	for t=1:15
		x(t) = t;
		Y=unitVec(X*W)';					% Do	
		deltaW = (Y_train - Y)';
		W = W + eta.*(deltaW*X)';			% Train
		param.W = W(1:(end-1));
		param.b = W(end);
		% Predict
		[Y_resu_train, Y_conf_train] 	= predict( X_train, param, feat);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param, feat);
		% Blance error for train and validiate
		err_bal_train(t)				= balanced_errate(Y_resu_train, Y_train)*100;
    	err_bal_valid(t)				= balanced_errate(Y_resu_valid, Y_valid)*100;
		% AUC error for train and validiate
		err_auc_train(t)				= auc(Y_resu_train.*Y_conf_train, Y_train)*100;
    	err_auc_valid(t)				= auc(Y_resu_valid.*Y_conf_valid, Y_valid)*100;
	end
	figure;
	title('Perceptron Training')
	subplot(1,2,1)
	plot(x,err_bal_train,x,err_bal_valid)
	ylabel('BER(%)')
	xlabel('Iterations')
	grid on;
	subplot(1,2,2)
	plot(x,err_auc_train,x,err_auc_valid)
	ylabel('AUC(%)')
	xlabel('Iterations')
	grid on ;
	idx_feat = feat;
