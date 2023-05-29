% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

clc
clear
close all

tic

%% Load data
positive_data = load('P_zero.mat');
Unlabel_data = load('U_zero.mat');
Unlabel_data = Unlabel_data.Unlabel_data;
positive_data = positive_data.positive_data;
flag=1;

rand('state',0);
m_p = size(positive_data,1);
m_u = size(Unlabel_data,1);


for iter = 1:10
    iter
    %% Random sampling
    m_p_d = size(positive_data,1);
    m_u = size(Unlabel_data,1);
    train_u_ind = randperm(m_u,m_p_d);
    train_sample_u = Unlabel_data(train_u_ind,:);
    Unlabel_data(train_u_ind,:)=[];
    k_kfold = 3;
    data = [positive_data; train_sample_u];
    train_data_size = size(data,1);
    cv = cvpartition(train_data_size,'k',k_kfold);
    eval_svm = zeros(k_kfold,4);
    %3-fold
    for k_f =1: cv.NumTestSets
        train_ind = cv.training(k_f);
        test_ind = cv.test(k_f);
        train_label = data(train_ind,1);
        train_data = data(train_ind,2:end-1);
        test_label = data(test_ind,1);
        test_data = data(test_ind,2:end-1);
        
        
        %% Reliable Negative Set Extraction
        [ genes, genes_label, P_train, Reliable_Negative, refined_unlabeld ] = RnExtraction( train_data, train_label );
        
        size_RN = size(Reliable_Negative,1);
        size_ru = size(refined_unlabeld,1);
        size_P_train = size(P_train,1);

        [ W_i_j ] = Gene_mahal_similarity_net( [P_train ; Reliable_Negative; refined_unlabeld ] ); % or Gene_similarity_net

        [ LP, LN, WN, train_label ] = MultiRankWalk( P_train, Reliable_Negative, refined_unlabeld, W_i_j, train_label );
        %% Train
         [model_wKNN,predicted_label_wknn,eval_wknn(k_f,:), prob_knn_train ] = WeightedKnn( P_train, LP, Reliable_Negative, LN, WN, train_label, train_data, train_label,3,flag);
         [model_tree,predicted_label_wtree,eval_wtree(k_f,:), prob_wtree_train ] = WeightedCART( P_train,LP, Reliable_Negative,LN,WN, train_label, train_data, train_label, 0);
         [model_svm, predicted_label_svm, eval_svm(k_f,:), prob_svm_train] = WeightedSVM( P_train,LP, Reliable_Negative,LN,WN, train_label, train_data, train_label);

 
        %% Test            
         [model_wKNN,predicted_label_wknn,eval_wknn(k_f,:), prob_knn_test ] = WeightedKnn( P_train, LP, Reliable_Negative, LN, WN, train_label, test_data, test_label,3,flag);
         [model_tree,predicted_label_wtree,eval_wtree(k_f,:), prob_wtree_test ] = WeightedCART( P_train,LP, Reliable_Negative,LN,WN, train_label, test_data, test_label, 0);
         [model_svm, predicted_label_svm, eval_svm(k_f,:), prob_svm_test] = WeightedSVM( P_train,LP, Reliable_Negative,LN,WN, train_label, test_data, test_label);

       
        %% Ensemble
        %perceptron
        %x_train = [prob_knn_train prob_wtree_train prob_svm_train]';
        %y_train = train_label';
        
        %net = newp(x_train,y_train);
        %net.trainParam.epochs = 500;
        %net.trainParam.goal = 0.001;
        %net = train(net,x_train,y_train);
        
        %Final prediction
        %x_test = [prob_knn_test prob_wtree_test prob_svm_test]';
        %out = sim(net,x_test);
        %eval_o(k_f,:) = Evaluation(test_label,out');
        %disp(eval_o*100)
        
        %or MLP    
        x_train = [prob_knn_train prob_wtree_train prob_svm_train]';
        y_train = train_label';
        
        net = newff(x_train,y_train,[6 1]);
        net.trainParam.epochs = 500;
        net.trainParam.goal = 0.0001;
        net.divideParam.trainRatio=1;
        net.divideParam.testRatio=0;
        net.divideParam.valRatio=0;
        net = train(net,x_train,y_train);
        
        % Final prediction
        x_test = [prob_knn_test prob_wtree_test prob_svm_test]';
        out = sim(net,x_test);
        y_out = round(out);
        eval_o(k_f,:) = Evaluation(test_label,y_out');
        disp(eval_o*100)
        
    end
    
    Evaluation_model_o(iter,:) = mean(eval_o);
    Evaluation_model_o(iter,:)
    Evaluation_model_svm(iter,:) = mean(eval_svm);
    Evaluation_model_svm(iter,:)
    Evaluation_model_wknn(iter,:) = mean(eval_wknn);
    Evaluation_model_wknn(iter,:)
    Evaluation_model_tree(iter,:) = mean(eval_wtree);
    Evaluation_model_tree(iter,:)
end
avg_Evaluation_model_svm= mean(Evaluation_model_svm);
avg_Evaluation_model_wknn = mean(Evaluation_model_wknn);
avg_Evaluation_model_o = mean(Evaluation_model_o);
avg_Evaluation_model_tree = mean(Evaluation_model_tree);
std_o = std(Evaluation_model_o, 0,1);
std_tree = std(Evaluation_model_tree, 0,1);
toc









