function [ cb,cnt,energy ] = readcb( cbfile,cntfile,cblen,dim)
%HISTOGRAM Summary of this function goes here
%   Detailed explanation goes here

    fp = fopen(cbfile,'rb');
    cb = zeros(cblen,dim);
    for i=1:(cblen)
        cb(i,:) = fread(fp,dim,'float');
    end
    
    cb = round(cb);
    
    fclose(fp);
    
    fp = fopen(cntfile,'rb');
    cnt = fread(fp,cblen,'int64');
    fclose(fp);
    

    energy = sum(cb.*cb,2);  
    
    [energy,I] = sort(energy);
    cnt = cnt(I);
    cb = cb(I,:);
    
end

