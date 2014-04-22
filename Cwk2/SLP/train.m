function [param,idx_feat]=train(X_train, Y_train, X_valid, Y_valid, feat)
	eta=1;
	X=X_train(:,feat);						% All features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);			% Offset on all input patterns
	features=size(X,2);
	W=zeros(features,1);			% Init weights to zero 


	for t=1:200
		x(t) = t;
		Y=unitVec(X*W)';	
		
		
		deltaW = (Y_train - Y)';
		W = W + eta*(deltaW*X)';



		param.W = W(1:(end-1));
		param.b = W(end);

		[Y_resu_train, Y_conf_train] 	= predict( X_train, param, feat);
    	[Y_resu_valid, Y_conf_valid] 	= predict( X_valid, param, feat);

		% Blance error for train and validiate
		err_bal_train(t)				= balanced_errate(Y_resu_train, Y_train);
    	err_bal_valid(t)				= balanced_errate(Y_resu_valid, Y_valid);
		
		% AUC error for train and validiate
		err_auc_train(t)				= auc(Y_resu_train.*Y_conf_train, Y_train);
    	err_auc_valid(t)				= auc(Y_resu_valid.*Y_conf_valid, Y_valid);

		if((err_bal_train(t)	== 0) && (err_auc_train(t)	== 1))
			break;
		end
	end


	figure;
	subplot(1,2,1)
	plot(x,err_bal_train,x,err_bal_valid)
	grid on;
	subplot(1,2,2)
	plot(x,err_auc_train,x,err_auc_valid)
	grid on ;

	j = 1;
	for i=length(param.W):-1:1
		if(param.W(i) == 0)
			param.W(i) = [];	% remove
		else
			idx_feat(j) = i;	% keep and log
			j = j + 1;
		end
	end
	idx_feat = fliplr(idx_feat);	% flip order
