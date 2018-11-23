function res = Pooling(im,t,f,window_shape,pool_type,stride,pad_method)
    poolreg ={'MAX','AVG'};
    poolregfunc = {'max','mean'};
    func_reg = containers.Map(poolreg,poolregfunc);
    switch nargin
        case 4
            res = PoolingByType(im,t,f,window_shape,[2,2],'SAME',@max);
        case 7
            try
                pool_func = func_reg(pool_type);
                res = PoolingByType(im,t,f,window_shape,stride,pad_method,str2func(pool_func));
            catch error_info
                disp(error_info);
                error("Unknown Pooling Type");
            end
        otherwise
            error("Input Parameters Not Match");
    end
end

function res = PoolingByType(im,t,f,window_shape,stride,pad_method,pool_func)
    [im_h,im_w,im_d] = size(im);
    channel_size = [im_h,im_w];
    
    pad_key = {'SAME','VALID'};
    pad_func_reg = {'ceil','floor'};
    pad_type_reg = containers.Map(pad_key,pad_func_reg);
    try
        pad_func = str2func(pad_type_reg(pad_method));
        out_size = pad_func((channel_size-window_shape)./stride)+1;
    catch pad_error
        disp(pad_error);
        error("Unknown Pad Type");
    end
    if strcmp(pad_method,'SAME')
        [im,channel_size] = PoolPadding(im,channel_size,im_d,out_size,stride,window_shape,t,f);
    end
    
    res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func);
end

function [res,pad_channel_size] = PoolPadding(im,channel_size,im_d,out_size,stride,window_shape,t,f)
    pad_channel_size = (out_size-1).*stride+window_shape;
    res = fi(zeros([pad_channel_size,im_d]),t,f);
    res(1:channel_size(1),1:channel_size(2),:) = im;
end

function res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func)
    pool_len = channel_size-window_shape+1;
    
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(1),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
      
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');
    
    tmp = tmp(:,pos(:));
    tmp1 = zeros(prod(window_shape),prod(out_size),im_d);
    for i =1:im_d
        tmp1(:,:,i)=(i-1)*prod(channel_size)+tmp;
    end
    
    res = permute(reshape(pool_func(im(tmp1)),[out_size,im_d]),[2,1,3]);
end

function res = PoolingOneChannel(im,channel_size,out_size,window_shape,stride,pool_func)
    pool_len = channel_size-window_shape+1;
    
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(1),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
    
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');
    tmp = tmp(:,pos(:));
    res = reshape(pool_func(im(tmp)),out_size)';
end