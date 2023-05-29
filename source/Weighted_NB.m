% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ eval_nb_trian ] = Weighted_NB( data_nb )
num_cl = 2;
[m n ] = size(data_nb);

D_pluse = zeros(1,4004);
D_minus = zeros(1,4004);
for i_nb = 1:size_P_train+size_u
    if G_new(i_nb) >= 0
        D_pluse = [D_pluse; data_nb(i_nb,:)];
        score_pluse(size(D_pluse,1)-1,:) = G_new(i_nb);
    elseif G_new(i_nb) < 0
        D_minus = [D_minus; data_nb(i_nb,:)];
        score_minus(size(D_minus,1)-1,:) = G_new(i_nb);
    end
end
D_pluse(1,:) = [];
D_minus(1,:) = [];
train_label_nb = [ones(size(D_pluse,1),1) ; zeros(size(D_minus,1),1)];

total_score = [score_pluse; score_minus];
score = (total_score - min(total_score))./(max(total_score)-min(total_score));
score_pluse = score(1:size(score_pluse,1),1) ;
score_minus = score(size(score_pluse,1)+1:end,1) ;
for f_i = 1: m
    p_fk_tmp(f_i,1) = sum(D_pluse(:,f_i) .* score_pluse)+1;
end
for f_i = 1: n
    p_fk_tmp(f_i,2) = sum(D_minus(:,f_i) .* score_minus)+1;
end
for f_i = 1: n
    p_fk(f_i,1) = p_fk_tmp(f_i,1)/sum(p_fk_tmp(f_i,:));
end
for f_i = 1: n
    p_fk(f_i,2) = p_fk_tmp(f_i,2)/sum(p_fk_tmp(f_i,:));
end
% feature selection
%          data_nb(:,any(isnan(p_fk),2))=[];
%          test_data(:,any(isnan(p_fk),2))=[];
%          P_train(:,any(isnan(p_fk),2))=[];
%          Reliable_Negative(:,any(isnan(p_fk),2))=[];
%          refined_unlabeld(:,any(isnan(p_fk),2))=[];
%          p_fk(any(isnan(p_fk),2),:)=[];
%          [m n ] = size(data_nb);

new_data = data_nb+1;

%% calculating P(y) for each class
for k=1:num_cl
    prior(k) = 0.5;
end
% train
for i = 1:m
    for cla = 1:num_cl
        tmp=0;
        for f_k = 1 : n
            %             tmp = tmp * data_nb(i,f_k) * p_fk(f_k,cla);
            tmp = tmp + log(new_data(i,f_k) * p_fk(f_k,cla));
        end
        liklihood = (tmp);
        p_train_data(i,cla) = log(prior(cla))+(liklihood);
    end
    p_train_data(i,:) = exp(p_train_data(i,:) );
    p_train_data(i,:) = p_train_data(i,:)/sum(p_train_data(i,:));
    %      p_train_data(i,cla) = exp(p_train_data(i,cla) );
end
win_train = max(p_train_data,[],1);
%     ynew(i,1) = find(pff==temp);
win_train(win_train==2) =0;
eval_nb_trian(k_f,:) = Evaluation(train_label_nb,win_train);

%test
for i = 1:size(test_data,1)
    for cla = 1:num_cl
        tmp=1;
        for f_k = 1 : n
            tmp = tmp + log(test_data(i,f_k)) + log(p_fk(f_i,cla));
            %             tmp = tmp * test_data(i,f_k) * p_fk(f_k,cla);
        end
        liklihood = (tmp);
        p_test_data(i,cla) = prior(cla)*(liklihood);
    end
    p_test_data(i,cla) = p_test_data(i,cla)/sum(p_test_data(i,:));
end
win_test = max(p_test_data,[],1);
%     ynew(i,1) = find(pff==temp);
win_test(win_test==2) =0;
eval_nb(k_f,:) = Evaluation(train_label_nb,win_test);


end

