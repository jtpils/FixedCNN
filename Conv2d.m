function res = Conv2d(im,ker,t,f,stride,padding)
    [im_h,im_w,im_d]=size(im);
    [k_h,k_w,k_in,k_out] = size(ker);
    
    window_shape = [k_h,k_w];
    channel_size = [im_h,im_w];
    
    if im_d~=k_in
        error("Map dimension and Kernel dimension don't match.");
    end
    
    pad_key = {'SAME','VALID'};
    pad_func_reg = {'ceil','floor'};
    pad_type_reg = containers.Map(pad_key,pad_func_reg);
    
    try
        pad_func = str2func(pad_type_reg(padding));
        out_size = [pad_func((channel_size-window_shape)./stride)+1,k_out];
    catch error_info
        disp(error_info);
        error("Unknown Padding Type!");
    end
    
    if strcmp(padding,'SAME')
        [im,channel_size]= ConvPadding(im,t,f);
    end
    res = fi(zeros([out_size,k_out]),t,f);
    for i=1:k_in
        one_ker = reshape(ker(:,:,i,:),[window_shape,k_out]);
        res = res + Conv2dOneChannel(im(:,:,i),one_ker,channel_size,out_size,window_shape,stride);
    end
end

function res = Conv2dOneChannel(im,one_ker,channel_size,out_size,window_shape,stride)
    conv_len = channel_size-window_shape+1;
    ker_stack = reshape(one_ker,out_size(3),[]);
    
    pos_one_col = repmat([1:stride(1):conv_len(1)],[out_size(1),1]);
    gap_every_col = repmat(stride(2)*conv_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
    
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');
    tmp = tmp(:,pos(:));
    
    conv_res = ker_stack*im(tmp);
    res = reshape(conv_res,out_size)';
end

function ConvPadding(im,t,f)
    
end
