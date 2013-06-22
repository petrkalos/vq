function [ splits,out ] = split( name,cbfile,cntfile,cblen,dim,n )
%SPLIT Summary of this function goes here
%   Detailed explanation goes here

    %cblen = str2num(cblen);
    %dim = str2num(dim);
    %n = str2num(n);
    
    [~, cnt, energy] = readcb(cbfile,cntfile,cblen,dim);
    
    s = sum(cnt);
    step = s/n;
    
    count = 0;
    j=1;
    splits = zeros(n-1,1);
    for i=1:length(cnt)
        
        if(count>step)
            splits(j) = i;
            j = j+1;
            count = 0;%count-step;
        end
        count = count + cnt(i);
    end
    
    cnt = log10(cnt);
    splits = [1 splits' length(cnt)];
    
    
    out = cell(length(splits-1),1);
    for i=1:length(splits)-1
        if(splits(i)~=0 && splits(i+1)~=0)
            out{i} = cnt(splits(i):splits(i+1)); 
        end
    end
 
    semilogx(cnt);
    for i=1:length(splits)
        hold on;
        line([splits(i) splits(i)],[0 max(cnt)],'Color','r');
    end
    
    disp(energy(splits));
    xlabel('Cluster');
    ylabel('Cluster Frequency log');
    title(name);
    fp = fopen('splits.bin','wb');
    fwrite(fp,energy(splits),'int64');
    fclose(fp);
end

