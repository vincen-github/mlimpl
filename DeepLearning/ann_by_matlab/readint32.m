function [getdata]=readint32(FID)
    data = [];
    for i = 1:4
            f=fread(FID,1);
            data = strcat(data,num2str(dec2base(f,2,8)));
    end
    getdata = bin2dec(data);
end
