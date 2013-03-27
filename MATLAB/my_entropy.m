function h = my_entropy(x)
    

    x = x(x>0);
    
    p = x/sum(x);
    
    h = sum(-p.*log2(p));
    
end


