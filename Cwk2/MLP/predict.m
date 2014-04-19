function [Y_resu, Y_conf] = predict(X_test, param, feat)	
	X=X_test(:,feat);
	patterns=size(X,1);
	features=size(X,2);

	hidden=zeros(features,1);
	Y_score=zeros(patterns,1);

	
	for i=1:patterns
		%	for j=1:features
		%		for k=1:features
		%			hidden(j) = hidden(j) + param.w_in(j,k)*X(i,j);
		%		end
		%	end
		%	for j=1:features
		%		Y_score(i) = Y_score(i) + param.w_out(j).*hidden(j);		
		%	end
		
		Y_score(i) = param.w_out*(tanh(param.w_in*X(i,:)')+param.b_in)+param.b_out;
	end
	Y_resu=unitVec(Y_score)';	
	Y_conf=abs(Y_score);
