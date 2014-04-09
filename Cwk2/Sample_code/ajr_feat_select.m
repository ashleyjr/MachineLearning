function idx=ajr_feat_select(X, Y, num)
	fval=Y'*X;
	[sval, si]=sort(-fval);
	idx=si(1:num);
