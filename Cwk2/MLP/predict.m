function [Y_resu, Y_conf] = predict(X_test, param, feat)	
	X=X_test(:,feat);
	size(X)
	patterns=size(X,1);
	features=size(X,2);

	hidden=zeros(features);
	Y_score=zeros(patterns);
	for i=1:patterns
		for j=1:features
			for k=1:features
				hidden(j) = hidden(j) + param.w_in(j,k).*X(j,k);
			end
		end
		for j=1:features
			Y_score(i) = Y_score(i) + param.w_out(j).*hidden(j);		
		end
	end
	

	Y_resu=sign(Y_score);
	Y_conf=abs(Y_score);
