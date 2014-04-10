function [param, idx_out]=SLP_train(X_train, Y_train, idx_in)
	alpha = 0.01;
	X=X_train(:,idx_in);		% Subset of features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	size(X)
	size(ones_row)
	X=horzcat(X,ones_row)
	features=size(X,2);
	patterns=size(X,1);
	W=rand(features,1)-0.5;	% Init weights between -0.5 and 0.5 with length of training patterns 	
	b=0;	
	for i=1:500
		Y=X*W;
		for j=1:patterns
			if(Y(j)>0)
				Y(j) = 1;
			else
				Y(j) = -1;
			end
		end
		errate(i)=balanced_errate(Y, Y_train);
		pick=ceil(rand(1)*patterns);
		if(Y(pick) == Y_train(pick))
			W = W;
		else 
			if(Y(pick) < Y_train(pick))
				W_new = W + alpha.*X(pick,:)';
			else
				W_new = W - alpha.*X(pick,:)';
			end
			Y_new=X*W_new;
			if(balanced_errate(Y_new, Y_train) < errate(i))
				W = W_new;
			end
		end
	end
	figure(2)
	plot(errate)
	param.W = W(2:end)
	param.b = W(end)
	idx_out=idx_in;
