% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ LP, LN, WN, train_label ] = MultiRankWalk(P_train, Reliable_Negative, refined_unlabeld, W_i_j, train_label )
     
size_P_train = size(P_train,1);
size_RN = size(Reliable_Negative,1);
size_ru = size(refined_unlabeld,1);
% tmp = size_P_train+RN;

% Multi_Rank_walk
%% step1
G_new=[];
classes = unique(train_label);
for randwalk_class = 1 : size(classes,1)
    if randwalk_class == 1
        P_0 = ones(size_P_train,1);% prior prob of Positive
        N_0 = zeros(size_RN,1);  %prior prob of Reliable_Negative
        U_0 = zeros(size_ru,1);
        G_0 = [P_0; N_0; U_0];
    else
        P_0 = zeros(size_P_train,1);%
        N_0 = ones(size_RN,1);  %prior prob of Reliable_Negative
        U_0 = zeros(size_ru,1);
        G_0 = [P_0; N_0; U_0];
    end
    G_0 = G_0/sum(G_0);
    pointer_RN = size(P_0,1)+size(N_0,1);
    %% step2
    alpha = 0.8;
    size_i_j = size(W_i_j,1);
    D = zeros(size_i_j,size_i_j);
    for i= 1:size_i_j
        D(i,i) = sum(W_i_j(i,:));
    end
    W_i_j = inv(D) * W_i_j ;
    %G_r
    G_new(:,randwalk_class) = G_0;
    G_old = ones(size(G_0,1),1);
    while sqrt(sum((G_new(:,randwalk_class) - G_old).^2)) > 10^ -6
        G_old = G_new(:,randwalk_class);
        G_new(:,randwalk_class) = (1-alpha) * W_i_j * G_old + alpha * G_0;
    end
end
[G_new class_win] = max(G_new,[],2);
train_label = class_win;
train_label(train_label==2) = 0;

%% step2
%refined_unlabeld->LP(likly positive), LN(likly negative), WN(weak negative)
clp=0;
cln=0;
cwn=0;
LP = [];
LN = [];
WN = [];
for i = 1:size_ru
    if G_new(pointer_RN+i,:) > (1-alpha)/size(G_0,1)
        clp = clp + 1;
        LP(clp,:) = refined_unlabeld(i,:);
    elseif G_new(pointer_RN+i,:) < -( 1-alpha)/size(G_0,1)
        cln= cln+1;
        LN (cln,:) = refined_unlabeld(i,:);
    else
        cwn = cwn+1;
        WN (cwn,:) = refined_unlabeld(i,:);
    end
end
end

