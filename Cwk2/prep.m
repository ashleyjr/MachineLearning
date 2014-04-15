clear
addpath('Common')
dataset={'arcene', 'dexter', 'dorothea', 'gisette', 'madelon'};
data_dir='Data'				 
for k=1:length(dataset) 
	% Data
	data_name=dataset{k};
    input_dir=[data_dir '/' upper(data_name)];
	input_name=[input_dir '/' data_name]
	fprintf('\n/|\\-/|\\-/|\\-/|\\ Loading and checking dataset %s /|\\-/|\\-/|\\-/|\\\n\n', upper(data_name));
	% Read
	p=read_parameters([input_name '.param'])
	fprintf('-- %s parameters and statistics -- \n\n', upper(data_name));
	print_parameters(p);
    fprintf('\n-- %s loading data --\n', upper(data_name));
    X_train=[]; X_valid=[]; X_test=[]; Y_train=[]; Y_valid=[]; Y_test=[];
    if fcheck([data_dir '/' data_name '.mat']), 
        load([data_dir '/' data_name]); 
    else
        fprintf('\n');
	    % Read the labels
	    Y_train=read_labels([input_name '_train.labels']);
	    Y_valid=read_labels([input_name '_valid.labels']);  
	    Y_test=read_labels([input_name '_test.labels']);   
	    % Read the data
	    X_train=matrix_data_read([input_name '_train.data'],p.feat_num,p.train_num,p.data_type);
	    X_valid=matrix_data_read([input_name '_valid.data'],p.feat_num,p.valid_num,p.data_type);
        X_test=matrix_data_read([input_name '_test.data'],p.feat_num,p.test_num,p.data_type);
        save([data_dir '/' data_name], 'X_train', 'X_valid', 'X_test', 'Y_train', 'Y_valid', 'Y_test');
    end
    fprintf('\n-- %s data loaded --\n', upper(data_name));
	% Check the labels
	check_labels(Y_train, p.train_num, p.train_pos_num);
    if ~isempty(Y_valid), check_labels(Y_valid, p.valid_num, p.valid_pos_num); end
    if ~isempty(Y_test), check_labels(Y_test, p.test_num, p.test_pos_num); end
	% Check the data
	check_data(X_train, p.train_num, p.feat_num, p.train_check_sum);
	check_data(X_valid, p.valid_num, p.feat_num, p.valid_check_sum);
	check_data(X_test, p.test_num, p.feat_num, p.test_check_sum);
	fprintf('\n-- %s data sanity checked --\n', upper(data_name));
end % Loop over datasets
