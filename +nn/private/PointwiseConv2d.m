% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below
%{
    This function is point-wise conv2d, maybe merged into Conv2d in the
    future. Because when convoluting an imgage with 1*1 filters, we needn't
    im2col to reorder image and multiplying image with filter value is
    more efficient. Thus, I try to write point-wise conv2d to deal with 1*1
    filters.
    TODO:
        1. stride~=[1,1]
        2. padding method support
%}
% 
function res = PointwiseConv2d(im,ker,t,f)
    [im_h,im_w,im_d] = size(im);
    [~,~,k_in,k_out] = size(ker);
    
    im_mat = reshape(im,[im_h*im_w,im_d]);
    ker_mat = reshape(ker,[k_in,k_out]);

%     res_mat = MultiCoreGEMM(im_mat,ker_mat);
    res_mat = FXPGEMMonGPU(im_mat,ker_mat);
    res = reshape(res_mat,[im_h,im_w,k_out]);
end