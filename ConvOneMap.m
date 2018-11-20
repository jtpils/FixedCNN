function o_map = ConvOneMap(im,ker,t,f)
    [im_h,im_w]=size(im);
    [k_h,k_w,k_c] = size(ker);
    im_mat = im2col(ZeroPadding(im,t,f),[k_h,k_w],'sliding');
    ker_mat = reshape(permute(ker,[3,1,2]),[k_c,numel(ker)/k_c]);
    
    stride = 1;
    out_h = ceil(im_h+2-k_h)/stride+1;
    out_w = out_h;
    o_map = reshape(flipud(rot90(ker_mat*im_mat)),[out_h,out_w,k_c]);
end