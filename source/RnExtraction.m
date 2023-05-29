% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ genes, genes_label, P_train, Reliable_Negative, refined_unlabeld ] = RnExtraction( train_data, train_label  )
RN = zeros(1,1);
counter = 0;
P_train = train_data((train_label==1),:);
U_train = train_data((train_label==0),:);
size_u = size(U_train,1);
size_P_train = size(P_train,1);
Pr = sum(P_train)/size_P_train; % positive represantative
Ave_dist = 0;
for each_gi_U = 1:size_u
    d(each_gi_U,1) = pdist2(Pr,U_train(each_gi_U,:) );
    Ave_dist = Ave_dist + ( d(each_gi_U,1)/size_u);
end
for each_gi_U = 1:size_u
    if d(each_gi_U,1)> Ave_dist
        counter = counter+1;
        RN(counter,1) = each_gi_U;
    end
end
Reliable_Negative = U_train(RN,:);
size_RN = size(Reliable_Negative,1);
refined_unlabeld = U_train;
refined_unlabeld(RN,:)=[];  %U-RN
size_ru = size(refined_unlabeld,1);

%% Co Training
temp_trian_data = [P_train;Reliable_Negative ];
temp_trian_label = [ones(size_P_train,1); zeros(size_RN,1)];
for loop=1: 50
    %loop
    size_ru = size(refined_unlabeld,1);
    U_prime_ind = randperm(size_ru,round(size_ru/10));
    U_prime = refined_unlabeld(U_prime_ind,:);
    refined_unlabeld(U_prime_ind,:)=[];
    model_KNN_1 = ClassificationKNN.fit(temp_trian_data(:,1:3000),temp_trian_label,'DistanceWeight', 'squaredinverse','Distance' , 'chebychev');
    %'mahalanobis'err , 'cosine', 'chebychev': best, 'correlation':worst,    'minkowski' :good,spearman:good; hamming:good,jaccard:good
    % chebychev:::> 'DistanceWeight', 'equal' :dont change, 'DistanceWeight',
    % 'inverse':little improve 0.02 , 'DistanceWeight', 'squaredinverse' :1% improve
    model_KNN_1.NumNeighbors = 3 ;
    %%Test
    [predicted_label_1, prob_wnn_1] = predict(model_KNN_1,U_prime(:,1:3000));
    confidence_prob_1 = abs(prob_wnn_1(:,1) - prob_wnn_1(:,2));
    
    model_KNN_2 = ClassificationKNN.fit(temp_trian_data(:,3001:end),temp_trian_label,'DistanceWeight', 'squaredinverse','Distance' , 'chebychev');
    model_KNN_2.NumNeighbors = 3 ;
    %%Test
    [predicted_label_2, prob_wnn_2] = predict(model_KNN_2,U_prime(:,3001:end));
    confidence_prob_2 = abs(prob_wnn_2(:,1) - prob_wnn_2(:,2));
    
    [same_ind same_label] = find(predicted_label_1 == predicted_label_2  );
    candidate_p = U_prime(predicted_label_1(same_ind,1)==1,:);
    candidate_n = U_prime(predicted_label_1(same_ind,1)==0,:);
    U_prime(predicted_label_1(same_ind,1)==1 | predicted_label_1(same_ind,1)==0 ,:)=[];
    %U_prime(predicted_label_1(same_ind,1)==0,:)=[];
    if size(candidate_n,1) == 0 || size(candidate_p,1) ==0
        select_n = 0;
        select_p = 0;
    elseif size(candidate_n,1)< 3 || size(candidate_p,1)==0
        select_n = size(candidate_n,1);
        %select_p = 1;
        select_p = 0;
    else
        select_n = size(candidate_n,1);
        %select_p = round(size(candidate_n,1)/3);
        select_p = 0;
    end
    [con_value_n con_ind_n] = sort(confidence_prob_1(predicted_label_1(same_ind,1)==0)+confidence_prob_2(predicted_label_1(same_ind,1)==0),'descend');
    [con_value_p con_ind_p] = sort(confidence_prob_1(predicted_label_1(same_ind,1)==1)+confidence_prob_2(predicted_label_1(same_ind,1)==1),'descend');
    
    temp_trian_data = [temp_trian_data;candidate_n(con_ind_n(1:select_n,1),:);candidate_p(con_ind_p(1:select_p,1),:)];
    candidate_p(con_ind_p(1:select_p,1),:)=[];
    candidate_n(con_ind_n(1:select_n,1),:)=[];
    temp_trian_label = [temp_trian_label;zeros(select_n,1); ones(select_p,1) ];
    refined_unlabeld = [refined_unlabeld; candidate_p;candidate_n;U_prime];        
end

Reliable_Negative = temp_trian_data(temp_trian_label==0,:);
temp_trian_data(temp_trian_label==0,:)=[];
genes = [temp_trian_data;Reliable_Negative; refined_unlabeld];
size_RN = size(Reliable_Negative,1);
size_ru = size(refined_unlabeld,1);
size_P_train = size(temp_trian_data,1);
P_train = temp_trian_data;
genes_label = [ones(size_P_train,1); zeros(size_ru+size_RN,1)];


end

