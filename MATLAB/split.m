function [ cntd,out ] = split( cnt,n )
%SPLIT Summary of this function goes here
%   Detailed explanation goes here

    s = sum(cnt);
    step = s/n;
    
    count = 0;
    j=1;
    cntd = zeros(n-1,1);
    for i=1:length(cnt)
        count = count + cnt(i);
        if(count>step)
            if(i>2 && abs(count-step)>abs((count-cnt(i-1))-step))
                cntd(j) = i-1;
            else
                cntd(j) = i;
            end
            j= j+1;
            count = 0;
        end
    end
    
    cnt = log10(cnt);
    
    cntd = [1 cntd' length(cnt)];
    
    
    out = cell(length(cntd-1));
    for i=1:length(cntd)-1
        out{i} = cnt(cntd(i):cntd(i+1));
        
    end
    
    semilogx(cnt);
    for i=1:length(cntd)
        hold on;
        line([cntd(i) cntd(i)],[0 max(out{i})],'Color','r');
    end
    
end

