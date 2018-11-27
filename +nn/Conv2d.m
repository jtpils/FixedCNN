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
%   If kernel size is 1*1, use PointwiseConv2d to accelerate calculation.
        res = PointwiseConv2d(im,ker,t,f);
    end
end

function res = Conv2dTensor(im,ker,im_d,k_out,channel_size,out_size,window_shape,stride)
%   Get element position of input feature map.
    tmp1 = GetElemPos(im_d,channel_size,out_size,window_shape,stride);

%   Reshape kernel and input feature map into im2col matrix
    ker_mat = reshape(ker,[im_d*prod(window_shape),k_out])';  
    im_mat =  reshape(permute(im(tmp1),[2,1,3]),[prod(out_size),prod(window_shape)*im_d])';
  
%   Calculate Conv2d result by GEMM (General Matrix Multiplication)
%    conv_tmp = ker_mat*im_mat;

    conv_tmp = MultiCoreGEMM(ker_mat,im_mat);
%   Reshape result matrix into tensor format to match the output shape  
    res = permute(reshape(conv_tmp',[fliplr(out_size),k_out]),[2,1,3]);
end