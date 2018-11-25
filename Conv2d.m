% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: Conv2d function

function res = Conv2d(im,ker,t,f,stride,padding)
    [im_h,im_w,im_d]=size(im);
    [k_h,k_w,k_in,k_out] = size(ker);
    
    window_shape = [k_h,k_w];
    channel_size = [im_h,im_w];
    
    if im_d~=k_in
        error("Map dimension and Kernel dimension don't match.");
    end
    if k_h>1 || k_w>1
        [im,out_size,channel_size]=PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,padding);
        res = Conv2dTensor(im,ker,im_d,k_out,channel_size,out_size,window_shape,stride); 
    else
        res = PointwiseConv2d(im,ker,t,f);
    end
end

function res = Conv2dTensor(im,ker,im_d,k_out,channel_size,out_size,window_shape,stride)
    pool_len = channel_size-window_shape+1;
    
    pos_one_col = repmat([1:stride(1):pool_len(1)],[out_size(2),1]);
    gap_every_col = repmat(stride(2)*pool_len(1)*[0:out_size(2)-1]',[1,out_size(1)]);
    pos = pos_one_col+gap_every_col;
      
    tmp = im2col(reshape([1:prod(channel_size)],channel_size),window_shape,'sliding');
    
    tmp = tmp(:,pos(:));
%     tmp1 = zeros(prod(window_shape),prod(out_size),im_d);
    
%     tmp1 = repmat(tmp,[1,im_d]);
%     tmp1 = kron(0:im_d-1,prod(channel_size))+tmp1;
%     tmp1 = reshape(tmp1,channel)
    [t1,t2] = size(tmp);
    tmp1 = repelem((0:im_d-1)*prod(channel_size),t1,t2)+repmat(tmp,1,im_d);
    tmp1 = reshape(tmp1,[t1,t2,im_d]);
%     
%     for i =1:im_d
%         tmp1(:,:,i)=(i-1)*prod(channel_size)+tmp;
%     end
    
    ker_mat = reshape(ker,[im_d*prod(window_shape),k_out])';
    
    im_mat =  reshape(permute(im(tmp1),[2,1,3]),[prod(out_size),prod(window_shape)*im_d])';
    
    conv_tmp = ker_mat*im_mat;
    res = permute(reshape(conv_tmp',[fliplr(out_size),k_out]),[2,1,3]);
end