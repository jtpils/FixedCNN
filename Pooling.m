% function res = Pooling(im,t,f,poolsize)
%     [im_h,im_w,im_d]=size(im);
%     b_h = poolsize(1);
%     b_w = poolsize(2);
%     block_rows = im_h/poolsize(1);
%     block_cols = im_w/poolsize(2);
%     im_block = mat2cell(im,b_h*ones(1,block_rows),b_w*ones(1,block_cols),[im_d]);
%     tmp = cellfun(@(x) max(reshape(x,b_h*b_w,im_d)),im_block,'UniformOutput',false);
%     tmp = cellfun(@(x) x.data,tmp,'UniformOutput',false);
%     res_r = cell2mat(tmp);
%     res = fi(res_r,t,f);
%     res = reshape(res',3,[]);
%     res = permute(reshape(res',16,16,3),[2,1,3]);
% end

function res = Pooling(im,t,f,poolsize,pool_type,poolstride,pad_method)
    switch nargin
        case 4
            res = MaxPooling(im,t,f,poolsize,[2,2],'SAME');
        case 7
            if strcmp(pool_type,'MAX')
                res = MaxPooling(im,t,f,poolsize,poolstride,pad_method);
            elseif strcmp(pool_type,'AVG')
                res = AVGPooling(im,t,f,poolsize,poolstride,pad_method);
            else
                error("Unknown Pooling Type");
            end
        otherwise
            error("Input Parameters Not Match");
    end
end

function res = MaxPooling(im,t,f,poolsize,poolstride,pad_method)
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
        
        for i=[1:im_d]
            res(:,:,i)=MAXPoolingOneChannel(impad(:,:,i),out_h,t,f,poolsize,poolstride);
        end
        
    elseif strcmp(pad_method,'VALID')
        out_h = floor((im_h-poolsize(1))/s_x)+1;
        out_w = floor((im_w-poolsize(2))/s_y)+1;
        res = fi(zeros(out_h,out_w,im_d),t,f);
        
        for i=[1:im_d]
            res(:,:,i)=MAXPoolingOneChannel(im(:,:,i),out_h,t,f,poolsize,poolstride);
        end
    else
        error("Unknown Output Type");
    end
end

function res = AVGPooling(im,t,f,poolsize,poolstride,pad_method)
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
        
        for i=[1:im_d]
            res(:,:,i)=AVGPoolingOneChannel(impad(:,:,i),out_h,t,f,poolsize,poolstride);
        end
        
    elseif strcmp(pad_method,'VALID')
        out_h = floor((im_h-poolsize(1))/s_x)+1;
        out_w = floor((im_w-poolsize(2))/s_y)+1;
        res = fi(zeros(out_h,out_w,im_d),t,f);
        
        for i=[1:im_d]
            res(:,:,i)=AVGPoolingOneChannel(im(:,:,i),out_h,t,f,poolsize,poolstride);
        end
    else
        error("Unknown Output Type");
    end
end

function res = PoolPadding(im,out_h,stride,window_w,t,f)
    [im_h,im_w,im_d]=size(im);
    padw = (out_h-1)*stride+window_w;
    res = fi(zeros(padw,padw,im_d),t,f);
    res(1:im_h,1:im_w,:)=im;
end

function res = MAXPoolingOneChannel(im,out_h,t,f,poolsize,poolstride)
    [im_h,im_w]=size(im);
    s_x = poolstride(1);
    s_y = poolstride(2);
    w_h = poolsize(1);
    w_w = poolsize(2);
    
    pool_len = im_h-w_h+1;
    pos = repmat([1:s_x:pool_len],[out_h,1])+repmat(s_x*pool_len*[0:out_h-1]',[1,out_h]);
    pool_col = max(im2col(im,[w_h,w_w],'sliding'));
    
    res = pool_col(pos(:));
    res = reshape(res,[out_h,out_h])';
end

function res = AVGPoolingOneChannel(im,out_h,t,f,poolsize,poolstride)
    [im_h,im_w]=size(im);
    s_x = poolstride(1);
    s_y = poolstride(2);
    w_h = poolsize(1);
    w_w = poolsize(2);
    
    pool_len = im_h-w_h+1;
    pos = repmat([1:s_x:pool_len],[out_h,1])+repmat(s_x*pool_len*[0:out_h],[1,out_h]);
    pool_col = max(im2col(im,[w_h,w_w],'sliding'));
    
    res = pool_col(pos(:));
    res = reshape(res,[out_h,out_h])';
end

