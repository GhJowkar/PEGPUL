function [ W_i_j ] = Gene_mahal_similarity_net( genes )
% data = P and U

A= cov(genes);
W = pinv(A);
D = mahaldistance(genes, genes, W);
% D= pdist(genes,'mahalanobis',W);
m = size(genes,1);
min_i_k = min(D,[],2);
max_i_k = max(D,[],2);

% parpool
for  i=1:m
   for j=i:m
       W_i_j(i,j) = 1-((D(i,j) - min_i_k(i,1))/ ( max_i_k(i,1) - min_i_k(i,1)) );      
       W_i_j(j,i) = W_i_j(i,j);
   end
end


end

