function res = Pooling(im,t,f,poolsize,pool_type,poolstride,pad_method)
    poolreg={'MAX','AVG'};
    poolregfunc = {'max','mean'};
    func_reg = containers.Map(poolreg,poolregfunc);
    switch nargin
        case 4
            res = PoolingByType(im,t,f,poolsize,[2,2],'SAME',@max);
        case 7
            try
                pool_func = func_reg(pool_type);
                res = PoolingByType(im,t,f,poolsize,poolstride,pad_method,str2func(pool_func));
            catch error_info
                disp(error_info);
                error("Unknown Pooling Type");
            end
        otherwise
            error("Input Parameters Not Match");
    end
end

function res = PoolingByType(im,t,f,poolsize,poolstride,pad_method,pool_func)
    [im_h,im_w,im_d]=size(im);
    s_x=poolstride(1);
    s_y=poolstride(2);
    
    w_h=poolsize(1);
    w_w=poolsize(2);
    
    output_key = {'SAME','VALID'};
    output_func = {'ceil','floor'};
    output_type_reg = containers.Map(output_key,output_func);
    try
        pad_func = str2func(output_type_reg(pad_method));
        out_h = pad_func((im_h-poolsize(1))/s_x)+1;
        out_w = pad_func((im_w-poolsize(2))/s_y)+1;
    catch pad_error
        disp(pad_error);
        error("Unknown Pad Type");
    end
    if strcmp(pad_method,'SAME')
        [im,im_h,im_w] = PoolPadding(im,im_h,im_w,im_d,out_h,s_x,w_h,t,f);
    end
    res = fi(zeros(out_h,out_w,im_d),t,f);    
    for i=1:im_d
        res(:,:,i)=PoolingOneChannel(im(:,:,i),im_h,im_w,out_h,poolsize,poolstride,pool_func);
    end
end

function res = PoolingOneChannel(im,im_h,im_w,out_h,poolsize,poolstride,pool_func)
    s_x = poolstride(1);
    s_y = poolstride(2);
    w_h = poolsize(1);
    w_w = poolsize(2);
    
    pool_len = im_h-w_h+1;
    pos = repmat([1:s_x:pool_len],[out_h,1])+repmat(s_x*pool_len*[0:out_h-1]',[1,out_h]);
    pool_col = pool_func(im2col(im,[w_h,w_w],'sliding'));
    
    tmp = pool_col(pos(:));
    res = reshape(tmp,[out_h,out_h])';
end

function [res,padh,padw] = PoolPadding(im,im_h,im_w,im_d,out_h,stride,window_w,t,f)
    padh = (out_h-1)*stride+window_w;
    padw = (out_h-1)*stride+window_w;
    res = fi(zeros(padw,padw,im_d),t,f);
    res(1:im_h,1:im_w,:)=im;
end