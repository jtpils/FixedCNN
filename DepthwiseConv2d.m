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
    pool_len = channel_size-window_shape+1;
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(2),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
      
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');

    tmp = tmp(:,pos(:));
    [t1,t2] = size(tmp);
    tmp1 = repelem((0:im_d-1)*prod(channel_size),t1,t2)+repmat(tmp,1,im_d);
    tmp1 = reshape(tmp1,t1,t2,[]);
    
    ker_mat = reshape(permute(ker,[1,2,4,3]),[prod(window_shape)*im_d,multiplier])';
    ker_cell = mat2cell(ker_mat,[multiplier],prod(window_shape)*ones(1,im_d));
    
    im_cell = mat2cell(reshape(im(tmp1),t1,[]),[prod(window_shape)],[prod(out_size)*ones(1,im_d)]);
    res_cell = cellfun(@mtimes,ker_cell,im_cell,'UniformOutput',false);
    
    res = fi(zeros([out_size,im_d*multiplier]),t,f);
    for i=1:im_d
        res(:,:,1+(i-1)*multiplier:i*multiplier)=reshape(res_cell{i}',[out_size,multiplier]);
    end
end