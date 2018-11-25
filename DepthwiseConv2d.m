function res = DepthwiseConv2d(im,ker,t,f,stride,padding_method)
    [im_h,im_w,im_d]=size(im);
    [k_h,k_w,k_in,multiplier] = size(ker);
    
    window_shape = [k_h,k_w];
    channel_size = [im_h,im_w];
    
    if im_d~=k_in
        error("Map dimension and Kernel dimension don't match.");
    end

    [im,out_size,channel_size] = PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,padding_method);
    res = DepthwiseConvTensor(im,ker,t,f,im_d,multiplier,channel_size,out_size,window_shape,stride);
end

function res = DepthwiseConvTensor(im,ker,t,f,im_d,multiplier,channel_size,out_size,window_shape,stride)
    res = fi(zeros([out_size,im_d]),t,f);
    pos = im2col(reshape(1:prod(channel_size),channel_size),window_shape,'sliding');
    
end