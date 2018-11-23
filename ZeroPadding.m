function impad = ZeroPadding(im,t,f)
    im_shape = size(im);
    impad = fi(zeros(im_shape+2),t,f);
%    impad = repmat(fi(0,t,f),im_shape+2);
    impad(2:end-1,2:end-1)=im;
end