function [entr,p] = context_stats( filename,num_of_categories,num_of_clusters )
%CONTEXT_STATS Summary of this function goes here
%   Detailed explanation goes here
    
    fp = fopen(filename,'rb');
    cnts = fread(fp,num_of_clusters*num_of_categories,'uint64');
    fclose(fp);
    
    cnts = reshape(cnts,num_of_clusters,num_of_categories)';
    
    cnts = sum(cnts);
    
    total_sum = sum(cnts);
    total_sum
    p = cnts/total_sum;
    
    entr = wentropy(p,'shannon')/log(2);
    
end

