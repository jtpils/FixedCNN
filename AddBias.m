function res = AddBias(im,bias,t,f)
    [im_h,im_w,im_d]=size(im);
    res = im;
    for i = [1:1:im_d]
        res(:,:,i) = im(:,:,i) + bias(i);
    end
end