function res = Pooling(im,t,f,window_shape,pool_type,stride,pad_method)
    poolreg={'MAX','AVG'};
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
    [im_h,im_w,im_d]=size(im);
    channel_size = [im_h,im_w];
    
    output_key = {'SAME','VALID'};
    output_func = {'ceil','floor'};
    output_type_reg = containers.Map(output_key,output_func);
    try
        pad_func = str2func(output_type_reg(pad_method));
        out_size = pad_func((channel_size-window_shape)./stride)+1;
    catch pad_error
        disp(pad_error);
        error("Unknown Pad Type");
    end
    if strcmp(pad_method,'SAME')
        [im,im_pad_size] = PoolPadding(im,channel_size,im_d,out_size,stride,window_shape,t,f);
    end
    res = fi(zeros([out_size,im_d]),t,f);    
    for i=1:im_d
        res(:,:,i)=PoolingOneChannel(im(:,:,i),im_pad_size,out_size,window_shape,stride,pool_func);
    end
end

function res = PoolingOneChannel(im,im_pad_size,out_size,window_shape,stride,pool_func)
    pool_len = im_pad_size-window_shape+1;
    
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(1),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
    
    tmp = im2col(reshape([1:prod(im_pad_size)],im_pad_size),window_shape,'sliding');
    tmp = tmp(:,pos(:));
    res = reshape(pool_func(im(tmp)),out_size)';
end

function [res,pad_out_size] = PoolPadding(im,channel_size,im_d,out_size,stride,window_shape,t,f)
    pad_out_size = (out_size-1).*stride+window_shape;
    res = fi(zeros([pad_out_size,im_d]),t,f);
    res(1:channel_size(1),1:channel_size(2),:)=im;
end