% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: Pooling function for 2d pooling of input tensor

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
    
    [im,out_size,channel_size] = PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,pad_method);
    
    res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func);
end

function res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func)
    pool_len = channel_size-window_shape+1;
    
%   此处有问题，不适用于长宽不相等的图像，待处理
%   计算滑动窗口对应的im2col后图像元素的标号，先计算标号，再提取数据创建矩阵，最后再进行矩阵
%   乘法是减少计算时间的核心，在必要的实际计算发生之前，不要轻易去创建fi对象和改变fi对象形状，
%   因为fi对象的创建和改变位置非常耗时。尽可能多使用标号来代替实际元素。
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(2),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
      
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');
    
%   此处有问题，不适用于长宽不相等的图像，待处理
    
%     tmp = tmp(:,pos(:));
%     tmp1 = zeros(prod(window_shape),prod(out_size),im_d);
%     for i =1:im_d
%         tmp1(:,:,i)=(i-1)*prod(channel_size)+tmp;
%     end

    tmp = tmp(:,pos(:));
    [t1,t2] = size(tmp);
    tmp1 = repelem((0:im_d-1)*prod(channel_size),t1,t2)+repmat(tmp,1,im_d);
    tmp1 = reshape(tmp1,[t1,t2,im_d]);
    
    res = permute(reshape(pool_func(im(tmp1)),[fliplr(out_size),im_d]),[2,1,3]);
end