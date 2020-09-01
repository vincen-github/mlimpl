function [y] = read_mnist_labels(filename)
    FID = fopen(filename,'r');

    MagicNumber=readint32(FID);
    NumberofImages=readint32(FID);

    y = zeros(NumberofImages,10);
    for i = 1:NumberofImages
                temp = fread(FID,1);
                y(i,temp+1) = 1;
    end
end