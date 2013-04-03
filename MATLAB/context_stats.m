function [h,entr,p,counters] = context_stats( filename,num_of_categories,num_of_clusters )
%CONTEXT_STATS Summary of this function goes here
%   Detailed explanation goes here
    
    fp = fopen(filename,'rb');
    counters = fread(fp,num_of_clusters*num_of_categories,'uint64');
    fclose(fp);
    
    counters = reshape(counters,num_of_clusters,num_of_categories)';
    
    row_sums = sum(counters,2); 
    
    sums2 = repmat(row_sums, 1, size(counters,2));
    row_probs = counters./sums2;
    
    for i=1:num_of_categories
       entr(i) = my_entropy(row_probs(i,:));
    end    
    
    %entr = my_entropy(row_probs);
    
    cat_sums = sum(counters');
    total_sums = sum(cat_sums);    
    
    p = cat_sums/total_sums;
    
    h = sum(entr.*p)/16;
    
end

