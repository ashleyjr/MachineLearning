function idx=feat_select(X)
	add = 1;
	for i=1:size(X,2)	
		if(rand() > 0.9)
			idx(add) = i;
			add = add + 1;
		end
	end
