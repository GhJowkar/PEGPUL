% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ model_tree,predicted_label_wtree,eval_wtree, prob_wtree_test ] = WeightedCART(  P_train, LP, Reliable_Negative, LN, WN,train_label, test_data, test_label,flag)
%% Weighted Tree
if  flag==0
    %         W_p = 11.471;
    %         W_lp =39.349	;
    %         W_rn = 12.96 ;
    %         W_ln =12.875;
    %         W_wn = 0.0625;
    W_p = 4.56;
    W_lp = 2.48	;
    W_rn = 27.225 ;
    W_ln = 57.09;
    W_wn = 29.18;
    training_weight_vector = ones(size(P_train,1),1)*W_p;
    training_weight_vector = [training_weight_vector ; ones(size(LP,1),1)*W_lp];
    training_weight_vector = [training_weight_vector ; ones(size(Reliable_Negative,1),1)*W_rn];
    training_weight_vector = [training_weight_vector ; ones(size(LN,1),1)*W_ln];
    training_weight_vector = [training_weight_vector ; ones(size(WN,1),1)*W_wn];
    data_wtr =  [P_train; LP;Reliable_Negative; LN; WN];
    model_tree = ClassificationTree.fit(data_wtr,train_label,'weights',training_weight_vector);
    %%Test
    [predicted_label_wtree, prob_wtree_test] = predict(model_tree,test_data);
    eval_wtree = Evaluation(test_label,predicted_label_wtree);
elseif flag ==1
    data_wtr =  [P_train; LP;Reliable_Negative; LN; WN];
    model_tree = ClassificationTree.fit(data_wtr,train_label,'SplitCriterion','deviance');
    %%Test
    [predicted_label_wtree, prob_wtree_test] = predict(model_tree,test_data);
    eval_wtree = Evaluation(test_label,predicted_label_wtree);
    
end
end

