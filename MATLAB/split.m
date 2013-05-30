function [ splits,out ] = split( cbfile,cntfile,cblen,dim,n )
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
        count = count + cnt(i);
        if(count>step)
            if(i>2 && abs(count-step)>abs((count-cnt(i-1))-step))
                splits(j) = i-1;
            else
                splits(j) = i;
            end
            j= j+1;
            count = 0;
        end
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
    
    fp = fopen('splits.bin','wb');
    fwrite(fp,energy(splits),'int64');
    fclose(fp);
    
end

