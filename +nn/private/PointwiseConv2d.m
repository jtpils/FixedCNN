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
    
    p = gcp('nocreate'); 
    if isempty(p)
        n = 0;
    else
        n = p.NumWorkers;
    end
    ker_h = ceil(k_out/n);
    
    ker_block = mat2cell(ker_mat,[ker_h*ones(1,n-1),k_out-ker_h*(n-1)],[k_in]);
    spmd
 %       tmp = ker_mat((labindex-1)*ker_h+1:labindex*ker_h,:)*im_mat;
        tmp = ker_block{labindex}*im_mat;
    end
    
    tmp2 = [tmp{1};tmp{2};tmp{3};tmp{4};tmp{5};tmp{6}];
    
    res = reshape(tmp2',[im_h,im_w,k_out]);
end