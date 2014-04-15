function [Y_resu, Y_conf] = SLP_predict(X_test, param, idx_feat, X_train, Y_train)
	Y_score=X_test(:,idx_feat)*param.W;
	b = ones(size(Y_score))*param.b;
	Y_score = Y_score + b;
	for j=1:size(Y_score)
		if(Y_score(j)>0)
			Y_score(j) = 1;
		else
			Y_score(j) = -1;
		end
	end
	Y_resu=sign(Y_score);
	Y_conf=abs(Y_score);
