function [ o ] = stats( filename )

    fp = fopen('context.bin','rb');
    buff = fread(fp,32768*256,'uint64');
    fclose(fp);
    
    cnt = reshape(buff,32768,256)';
    
    context = sum(cnt(:,:)');

    context_sum = sum(context);
    
    context_p = context/context_sum;
    
    o = context_p;
end

