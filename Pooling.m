function res = Pooling(im,t,f,poolsize,pool_type,poolstride,pad_method)
    switch nargin
        case 4
            res = PoolingByType(im,t,f,poolsize,[2,2],'SAME',@max);
        case 7
            if strcmp(pool_type,'MAX')
                res = PoolingByType(im,t,f,poolsize,poolstride,pad_method,@max);
            elseif strcmp(pool_type,'AVG')
                res = PoolingByType(im,t,f,poolsize,poolstride,pad_method,@mean);
            else
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
    
    if strcmp(pad_method,'SAME')
        out_h = ceil((im_h-poolsize(1))/s_x)+1;
        out_w = ceil((im_w-poolsize(2))/s_y)+1;
        res = fi(zeros(out_h,out_w,im_d),t,f);
        impad = PoolPadding(im,out_h,s_x,w_h,t,f);   
    elseif strcmp(pad_method,'VALID')
        out_h = floor((im_h-poolsize(1))/s_x)+1;
        out_w = floor((im_w-poolsize(2))/s_y)+1;
        res = fi(zeros(out_h,out_w,im_d),t,f);
        impad = im;
    else
        error("Unknown Output Type");
    end
    
    for i=1:im_d
        res(:,:,i)=PoolingOneChannel(impad(:,:,i),out_h,t,f,poolsize,poolstride,pool_func);
    end
end

function res = PoolingOneChannel(im,out_h,t,f,poolsize,poolstride,pool_func)
    [im_h,im_w]=size(im);
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

function res = PoolPadding(im,out_h,stride,window_w,t,f)
    [im_h,im_w,im_d]=size(im);
    padw = (out_h-1)*stride+window_w;
    res = fi(zeros(padw,padw,im_d),t,f);
    res(1:im_h,1:im_w,:)=im;
end