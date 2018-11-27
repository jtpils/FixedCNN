% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below
%{
    NOTE: DepthwiseConv2d don't support stride x ~= stride y for now as
    Tensorflow do.
%}

function res = DepthwiseConv2d(im,ker,t,f,stride,padding_method)
    [im_h,im_w,im_d]=size(im);
    [k_h,k_w,k_in,multiplier] = size(ker);
    
    window_shape = [k_h,k_w];
    channel_size = [im_h,im_w];
    
    if im_d~=k_in
        error("Map dimension and Kernel dimension don't match.");
    end
    if stride(1)~=stride(2)
        error("Current implementation only supports equal length strides in the row and column dimensions as TF do.");
    end

    [im,out_size,channel_size] = PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,padding_method);
    res = DepthwiseConvTensor(im,ker,t,f,im_d,multiplier,channel_size,out_size,window_shape,stride);
end

function res = DepthwiseConvTensor(im,ker,t,f,im_d,multiplier,channel_size,out_size,window_shape,stride)
%   Get element position of input feature map.
    im_pos = GetElemPos(im_d,channel_size,out_size,window_shape,stride);

%   Reshape kernel and input feature map into im2col cell
    ker_mat = reshape(permute(ker,[1,2,4,3]),[prod(window_shape),im_d*multiplier])';
    ker_cell = mat2cell(ker_mat,ones(1,im_d)*multiplier,prod(window_shape))';
    im_cell = mat2cell(reshape(im(im_pos),prod(window_shape),[]),[prod(window_shape)],[prod(out_size)*ones(1,im_d)]);

%   Calculate Conv2d result by GEMM (General Matrix Multiplication)

%   res_cell = cellfun(@MultiCoreGEMM,ker_cell,im_cell,'UniformOutput',false);

%   If Multi-Core mode is on, it will calculate GEMM with parfor otherwise it will calculate by cellfun locally. 
    num_core = GetCurrentCore();
    if num_core>0
        res_cell = cell(1,im_d);
        parfor i=1:im_d
            res_cell{i}=ker_cell{i}*im_cell{i};
        end
    else
        res_cell = cellfun(@mtimes,ker_cell,im_cell,'UniformOutput',false);
    end
    
%   Reshape result cell into tensor format to match the output shape
    res = fi(zeros([out_size,im_d*multiplier]),t,f);
    for i=1:im_d
        ch_res = permute(reshape(res_cell{i}',[fliplr(out_size),multiplier]),[2,1,3]);
        res(:,:,1+(i-1)*multiplier:i*multiplier)=ch_res;
    end
end