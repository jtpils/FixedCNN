function impad = ZeroPadding(im,t,f)
    im_shape = size(im);
    impad = repmat(fi(0,t,f),im_shape+2);
    impad(2:end-1,2:end-1)=im;
end