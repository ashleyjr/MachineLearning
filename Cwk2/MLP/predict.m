function [Y_resu, Y_conf] = predict(X_test, param, feat)	
	X=X_test(:,feat);
	hiddens=size(param.w_in,1)

	for j=1:hiddens
		Y_hidden(j,:) = tanh(X*param.w_in(j,:)'+param.b_in(j));
	end
	Y_score = unitVec(Y_hidden'*param.w_out')+param.b_out;
	Y_resu=unitVec(Y_score)';	
	Y_conf=abs(Y_score)';
