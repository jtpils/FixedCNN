function impad = ZeroPadding(im,t,f)
   im_shape = size(im);
%    impad = fi(zeros(im_shape+2),t,f);
%     impad = padarray(im.data,[1,1],0,'both');
%     impad = fi(impad,t,f);
%    impad = repmat(fi(0,t,f),im_shape+2);
%    impad(2:end-1,2:end-1)=im;
    
    addpad = zeros(1,length(im_shape));
    addpad(1:2)=[2,2];
    impad = fi(zeros(im_shape+addpad),t,f);
    impad(2:end-1,2:end-1,:)=im;
end