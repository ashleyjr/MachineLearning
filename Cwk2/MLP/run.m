clear
addpath('../Common')
dataset={'arcene', 'dexter', 'dorothea', 'gisette', 'madelon'};
dataset={'dexter'};
%dataset={'dexter', 'dorothea','gisette'};
method=('MLP')
where_my_data_is='../';            						% This is the path to your data and results are
data_dir=[where_my_data_is 'Data'] 						% Wehre you put the five data directories dowloaded.
output_dir=[where_my_data_is 'Results/' method] 		% The outputs of a given method.
status=mkdir(where_my_data_is, ['Results/' method]);
zip_dir=[where_my_data_is 'Zipped']; 					% Zipped files ready to go!
status=mkdir(where_my_data_is, 'Zipped');
    

for k=1:length(dataset)

    % Input and output directories 
	data_name=dataset{k};
    input_dir=[data_dir '/' upper(data_name)];
	input_name=[input_dir '/' data_name]
	output_name=[output_dir '/' data_name]
	fprintf('\n/|\\-/|\\-/|\\-/|\\ Loading and checking dataset %s /|\\-/|\\-/|\\-/|\\\n\n', upper(data_name));

	% Data parameters and statistics
	p=read_parameters([input_name '.param'])
	fprintf('-- %s parameters and statistics -- \n\n', upper(data_name));
	print_parameters(p);

    % Read the data
    fprintf('\n-- %s loading data --\n', upper(data_name));
    X_train=[]; X_valid=[]; X_test=[]; Y_train=[]; Y_valid=[]; Y_test=[];
   	load([data_dir '/' data_name]); 
    fprintf('\n-- %s data loaded --\n', upper(data_name));
    
	% Try some method
    fprintf('\n-- Begining... --\n\n');
	idx = feat_select(X_train,Y_train,30);

	[params,idx_feat] = train( X_train, Y_train, X_valid, Y_valid, idx);


	% Classifier has been trained, prediction only
	[Y_resu_train, Y_conf_train] 	= predict( X_train, params,	idx	);
    [Y_resu_valid, Y_conf_valid] 	= predict( X_valid, params,	idx	);
    %[Y_resu_test, Y_conf_test] 		= predict( X_test, 	params,	idx_feat	);


	% Blance error for train and validiate
	errate_train					= balanced_errate(Y_resu_train, Y_train);
    errate_valid					= balanced_errate(Y_resu_valid, Y_valid);
	
	% AUC error for train and validiate
	auc_train						= auc(Y_resu_train.*Y_conf_train, Y_train);
    auc_valid						= auc(Y_resu_valid.*Y_conf_valid, Y_valid);

	% User output	
	fprintf('Training set: errate= %5.2f%%, auc= %5.2f%%\n', errate_train*100, auc_train*100);
    fprintf('Validation set: errate= %5.2f%%, auc= %5.2f%%\n', errate_valid*100, auc_valid*100);



	% Write out the results 
	% --- Note: the class predictions (.resu files) are mandatory.
	% --- Please also provide the confidence value when available, this will
	% --- allow us to compute ROC curves. A confidence values can be the absolute
	% --- values of a discriminant value, it does not need to be normalized
	% --- to resemble a probability.
	%save_outputs([output_name '_train.resu'], Y_resu_train);
	%save_outputs([output_name '_valid.resu'], Y_resu_valid);
	%save_outputs([output_name '_test.resu'], Y_resu_test);
    %save_outputs([output_name '_train.conf'], Y_conf_train);
	%save_outputs([output_name '_valid.conf'], Y_conf_valid);
	%save_outputs([output_name '_test.conf'], Y_conf_test);
	%save_outputs([output_name '.feat'], idx_feat);
	%fprintf('\n-- %s results saved, see %s* --\n', upper(data_name), output_name);

end % Loop over datasets

% Zip the archive so it is ready to go!
zip([zip_dir '/' method], ['Results/' method], where_my_data_is);
fprintf('\n-- %s zip archive created, see %s.zip --\n', upper(data_name), [zip_dir '/' method]);



