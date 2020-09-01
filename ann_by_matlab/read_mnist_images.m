function [x] = mnist(filename)
    FID = fopen(filename,'r');

    MagicNumber=readint32(FID);
    NumberofImages=readint32(FID);
    rows=readint32(FID);
    colums=readint32(FID);
  
    x = zeros(NumberofImages,rows*colums);
    for i = 1:NumberofImages
                temp = fread(FID,(rows*colums), 'uchar');
                x(i,:) = temp';
    end

%     for i = 1:64
%         subplot(8,8,i);
%         imshow(reshape(x(i,:),28,28)');
%     end
end
    
    