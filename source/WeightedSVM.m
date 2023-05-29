% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ model_svm, predicted_label_svm, eval_svm, prob_svm_test ] = WeightedSVM (  P_train, LP, Reliable_Negative, LN, WN,train_label, test_data, test_label)
%% Weighted SVM
C = 741.84;
W_p = 1.86;
W_lp =2.8925	;
W_rn = 1.5775 ;
W_ln =1.10875;
W_wn = 1.395;
%         mahal dist
%         667.600000000000	1.94500000000000	2.18250000000000	1.23750000000000	1.83750000000000	0.125000000000000

% format [C_final, W_p, W_lp, W_ln, W_rn,  W_wn ]
%         [667.6 ,1.945 ,1.75 ,2.98625 ,2.24875 ,1]
%cosin simil 83%
%[ 839.200000000000	1.60000000000000	1.37500000000000 0.900000000000000	0.687500000000000	1]
%cosin avg
%[782.520000000000	1.93000000000000	3.37125000000000 1.33000000000000	1.03125000000000	1]
% train with fix value
% chebyshev sim
%         [C_final, W_p, W_lp, W_ln, W_rn,  W_wn ];
%         [741.84		1.86		2.8925		1.10875		1.5775	1.395	]


train_data_final =  [P_train; LP;Reliable_Negative; LN; WN];
train_label_final = [ones(size([P_train],1),1); zeros(size([LP;Reliable_Negative; LN; WN],1),1)];
option = [' -c '  num2str(C) ' -b 1' ];
training_weight_vector = ones(size(P_train,1),1)*W_p;
training_weight_vector = [training_weight_vector ; ones(size(LP,1),1)*W_lp];
training_weight_vector = [training_weight_vector ; ones(size(Reliable_Negative,1),1)*W_rn];
training_weight_vector = [training_weight_vector ; ones(size(LN,1),1)*W_ln];
training_weight_vector = [training_weight_vector ; ones(size(WN,1),1)*W_wn];

[ model_svm] = svmtrain(training_weight_vector, train_label, train_data_final,option); %RBF kernel exp(-0.5|u-v|^2), C=10
%Test
[predicted_label_svm, accuracy, prob_svm_test] = svmpredict(test_label, test_data, model_svm , '-b 1');
prob_svm_test = [prob_svm_test(:,2) prob_svm_test(:,1)];
eval_svm = Evaluation(test_label,predicted_label_svm);



end

