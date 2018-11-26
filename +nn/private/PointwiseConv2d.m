% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below
%{
    This function is point-wise conv2d, maybe merged into Conv2d in the
    future. Because when convoluting an imgage with 1¡Á1 filters, we needn't
    im2col to reorder image and multiplying image with filter value is
    more efficient. Thus, I try to write point-wise conv2d to deal with 1¡Á1
    filters.
%}
% 
function res = PointwiseConv2d(im,ker,t,f)
    [im_h,im_w,im_d] = size(im);
    [k_h,k_w,k_in,k_out] = size(ker);
    
    im_mat = reshape(im,[im_h*im_w,im_d])';
    ker_mat = reshape(ker,[k_in,k_out])';
    
    tmp = ker_mat*im_mat;
    res = reshape(tmp',[im_h,im_w,k_out]);
end

% function res = PointwiseConv2d(im,ker,t,f)
%     [im_h,im_w,im_d] = size(im);
%     [k_h,k_w,k_in,k_out] = size(ker);
%     
%     res = fi(zeros(im_h,im_w,k_out),t,f);
%     for i=1:k_out
%         for j=1:im_d
%             res(:,:,i)=res(:,:,i)+squeeze(ker(:,:,j,i));
%         end
%     end
%     
%     im_mat = reshape(im,[im_h*im_w,im_d])';
%     ker_mat = reshape(ker,[k_in,k_out])';
%     
%     tmp = ker_mat*im_mat;
%     res = reshape(tmp',[im_h,im_w,k_out]);
% end