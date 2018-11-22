function o_map = ConvOneMap(im,ker,t,f)
    [im_h,im_w]=size(im);
    [k_h,k_w,k_in,k_c] = size(ker);
    ker = reshape(ker,[k_h,k_w,k_c]);
    
    im_pad = ZeroPadding(im,t,f);
    pos_mat = im2col(reshape([1:numel(im_pad)],size(im_pad)),[k_h,k_w],'sliding');
    im_mat = im_pad(pos_mat);
    ker_mat = reshape(permute(ker,[3,1,2]),[k_c,numel(ker)/k_c]);
    
    stride = 1;
    out_h = ceil(im_h+2-k_h)/stride+1;
    out_w = out_h;
    conv_col = ker_mat*im_mat;
    o_map = reshape(flipud(rot90(conv_col)),[out_h,out_w,k_c]);
end