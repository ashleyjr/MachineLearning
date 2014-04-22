function [param,feat_out]=train(X_train, Y_train, X_valid, Y_valid, feat_in)
	eta=1;
	X=X_train(:,feat_in);						% Subset of features
	patterns=size(X,1);
	features=size(X,2);
	ones_row=ones(patterns,1);
	X=horzcat(X,ones_row);					% Offset on all input patterns
	w_in=ones(features,features+1)-0.5;		% Weights from input to each hidden layer node
	w_out=rand(1,features+1,1)-0.5;			% Weights from hidden layer to output 


	rec = 1
	for i=1:100	
		disp(i)

		param.w_out = w_out(1:(end-1));
		param.b_out = w_out(end);
		param.w_in = w_in(:,1:(end-1));
		param.b_in = mean(w_in(:,end));
		
		[resu, conf] = predict(X_train,param,feat_in);

		deltaW = eta*((Y_train - resu)'*X);
		w_out = w_out - deltaW;

		err(rec)=balanced_errate(resu, Y_train);
		rec = rec + 1;

		%for j=1:features
		%	[resu, conf] = predict(X_train,param,feat_in);	
		%	deltaW = eta*((Y_train - resu)'*X);
		%	w_in(j,:) = w_in(j,:) - deltaW;
		%	err(rec)=balanced_errate(resu, Y_train);
		%	rec = rec + 1;
		%end
	end

	figure;
	plot(err)
	
	feat_out = feat_in;
