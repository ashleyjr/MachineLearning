function [param,idx_feat]=train(X_train, Y_train, X_valid, Y_valid, feat)
	eta=1e-4;
	X=X_train(:,feat);						% All features
	patterns=size(X,1);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	features=size(X,2);
	W=rand(features,1)-0.5;					% Init weights to zero 
	for t=1:10
		x(t) = t;
		Y=unitVec(X*W)';					% Do	
		deltaW = (Y_train - Y)';
		W = W + eta.*(deltaW*X)';			% Train
	end
	param.W = W(1:(end-1));					% Split
	param.b = W(end);
	idx_feat = feat;	
