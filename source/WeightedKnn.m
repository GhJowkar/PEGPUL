% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [model_KNN,predicted_label_knn,eval_knn,prob_knn_test ] = WeightedKnn(  P_train, LP, Reliable_Negative, LN, WN, train_label, test_data, test_label,NumNeighbor,flag)
if  flag==0
    %% Weighted KNN
    W_p = 0.44625;
    W_lp =0.03125	;
    W_rn = 2.7334 ;
    W_ln =1.83;
    W_wn = 0;
    training_weight_vector = ones(size(P_train,1),1)*W_p;
    training_weight_vector = [training_weight_vector ; ones(size(LP,1),1)*W_lp];
    training_weight_vector = [training_weight_vector ; ones(size(Reliable_Negative,1),1)*W_rn];
    training_weight_vector = [training_weight_vector ; ones(size(LN,1),1)*W_ln];
    training_weight_vector = [training_weight_vector ; ones(size(WN,1),1)*W_wn];
    data_kwnn = [P_train; LP;Reliable_Negative; LN; WN];
    model_KNN = ClassificationKNN.fit(data_kwnn,train_label,'W',training_weight_vector,'DistanceWeight', 'squaredinverse','Distance' ,'chebychev');
    model_KNN.NumNeighbors = NumNeighbor ;
    %%Test
    [predicted_label_knn, prob_knn_test] = predict(model_KNN,test_data);
    eval_knn = Evaluation(test_label,predicted_label_knn);
elseif flag==1
    data_kwnn = [P_train; LP;Reliable_Negative; LN; WN];
    model_KNN = ClassificationKNN.fit(data_kwnn,train_label,'DistanceWeight', 'squaredinverse','Distance' ,'chebychev');
    model_KNN.NumNeighbors = NumNeighbor ;
    %%Test
    [predicted_label_knn, prob_knn_test] = predict(model_KNN,test_data);
    eval_knn = Evaluation(test_label,predicted_label_knn);
end

end

