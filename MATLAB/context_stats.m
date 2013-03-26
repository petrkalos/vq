function [entr,p,test] = context_stats( filename,num_of_categories,num_of_clusters )
%CONTEXT_STATS Summary of this function goes here
%   Detailed explanation goes here
    
    fp = fopen(filename,'rb');
    cnts = fread(fp,num_of_clusters*num_of_categories,'uint64');
    fclose(fp);
    
    cnts = reshape(cnts,num_of_clusters,num_of_categories)';
    
    sums = sum(cnts);
    
    total_sum = sum(sums);
    
    p = sums/total_sum;
    
    test = sums;
    
    entr = wentropy(p,'shannon')/log(2);
    
end

