function [param,idx_feat]=train(X_train, Y_train, X_valid, Y_valid, feat)
	eta=1e-2;
	X=X_train(:,feat);						% All features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	features=size(X,2);
	W=rand(features,1);					% Init weights to zero 
	for t=1:30
		x(t) = t;
		Y=tanh(X*W);					% Do	
		deltaW = (Y_train - Y)';
		W = W + eta.*(deltaW*X)';			% Train
	end
	param.W = W(1:(end-1));
	param.b = W(end);
	rec = 1;
	idx_feat = feat;
	test = abs(param.W);
	for i=1:5
		[C,I] = min(test);
		idx_feat(I) = [];
		test(I) = [];
	end
