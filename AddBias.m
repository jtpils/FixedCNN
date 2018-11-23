function res = AddBias(im,bias,t,f)
    [im_h,im_w,im_d]=size(im);    
    tmp = reshape(repmat(bias,[im_h*im_w,1]),[im_h,im_w,im_d]);
    res = im + tmp;
end