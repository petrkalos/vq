function [ output ] = b16to8( filename )
    %B16TO8 Summary of this function goes here
    %   Detailed explanation goes here

    fp = fopen(filename,'rb');
    yuv = fread(fp,260*720*480*3/2,'int16');
    fclose(fp);
    
    fp = fopen('newfile.yuv','wb');
    fwrite(fp,yuv,'uint8');
    fclose(fp);
    
    output = yuv;
end

