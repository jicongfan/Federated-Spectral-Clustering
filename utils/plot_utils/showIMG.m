function flag = showIMG(x, shape)
% read and show the image with the vector x

try
    imshow(reshape(x, shape));
    flag = 1;
catch
    flag = 0;
end

end