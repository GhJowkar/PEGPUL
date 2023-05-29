% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function [ W_i_j ] = Gene_similarity_net( genes )
% parpool
% data = P and U
m = size(genes,1);
d = zeros(m,m);
d=squareform(pdist(genes)); % pairwise distances, n-by-n matrix
min_i_k = min(d,[],2);
max_i_k = max(d,[],2);
% parpool
for  i=1:m
   for j=i:m
       W_i_j(i,j) = 1-((d(i,j)- min_i_k(i,1))/ ( max_i_k(i,1) - min_i_k(i,1)) );      
       W_i_j(j,i) = W_i_j(i,j);
   end
end


end

