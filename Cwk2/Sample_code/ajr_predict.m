function [Y_resu, Y_conf] = ajr_predict(X_test, param, idx_feat, X_train, Y_train)
	Y_score=X_test(:,idx_feat)*param.W'+param.b;
	Y_resu=sign(Y_score);
	Y_conf=abs(Y_score);
