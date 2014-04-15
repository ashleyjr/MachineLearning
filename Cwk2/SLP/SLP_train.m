function [param, idx_out]=SLP_train(X_train, Y_train, idx_in)
	mutation = 0.25;
	X=X_train(:,idx_in);		% Subset of features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);
	features=size(X,2);
	patterns=size(X,1);
	



	bestW = zeros(features,1);
	bestErrate = 1;
	for i=1:10
		W=rand(features,1)-0.5;	% Init weights between -0.5 and 0.5 with length of training patterns 	
		b=0;	
		for j=1:300
			Y=unitVec(X*W)';
			pick=ceil(rand(1)*features);					% Random feature
			W_new = W;
			if(rand(1) > 0.5)								% mutate
				W_new(pick) = W_new(pick) + mutation;
			else 
				W_new(pick) = W_new(pick) - mutation;
			end	
			Y_new=X*W_new;
			if(balanced_errate(Y_new, Y_train) < balanced_errate(Y, Y_train))	% Keep ?
				W = W_new;
			end
		end
		Y=unitVec(X*W)';
		errate=balanced_errate(Y, Y_train);
		if(errate < bestErrate)
			bestErrate = errate;
			bestW = W;
		end
	end
	
	
	param.W = W(1:(end-1));
	param.b = W(end);
	idx_out=idx_in;
