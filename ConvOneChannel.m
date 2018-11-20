function out_map = ConvOneChannel(im,ker,t,f)
    im_mat = im2col(ZeroPadding(im,t,f),size(ker),'sliding');
    ker_mat = reshape(ker,[1,numel(ker)]);
    
    [im_h,im_w]=size(im);
    [ker_h,ker_w]=size(ker);
    stride = 1;
    out_h = ceil(im_h+2-ker_h)/stride+1;
    out_w = out_h;
    out_map = reshape(ker_mat*im_mat,[out_h,out_w]);
end