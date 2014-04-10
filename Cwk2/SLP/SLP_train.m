function [param, idx_out]=SLP_train(X_train, Y_train, idx_in)
	X=X_train(:,idx_in);		% Subset of features
	features=size(X,2);
	patterns=size(X,1);
	W=rand(features,1)-0.5;	% Init weights between -0.5 and 0.5 with length of training patterns 	
	
	for i=1:1000
		Y=X*W
		pick=ceil(rand(1)*patterns);
		if(Y(pick) > Y_train(pick))
			W = W + X(pick,:)';
		else
			W = W - X(pick,:)';
		end
	end

	param.W = W;
	param.b = 0;
	idx_out=idx_in;
