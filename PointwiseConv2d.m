% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below
%{
    This function is point-wise conv2d, maybe merged into Conv2d in the
    future. Because when convoluting an imgage with 1��1 filters, we needn't
    im2col to reorder image and multiplying image with filter value is
    more efficient. Thus, I try to write point-wise conv2d to deal with 1��1
    filters.
%}
function res = PointwiseConv2d(im,ker,t,f,stride)
    [im_h,im_w,im_d] = size(im);
    [k_h,k_w,k_in,k_out] = size(ker);
    
    im = im(1:stride(1):im_h,1:stride(2):im_w,:);
    
    
    
end