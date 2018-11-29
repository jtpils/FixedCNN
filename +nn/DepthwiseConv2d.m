% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below
%{
    NOTE: DepthwiseConv2d don't support stride x ~= stride y for now as
    Tensorflow do.
    If the MultiCore mode is on, this function can get about 2 times
    speedup on 6 Cores Intel Core i5-8400 CPU.
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
    im_mat = reshape(im(im_pos),prod(window_shape),[]);

%   Calculate Conv2d result by GEMM (General Matrix Multiplication)
%   If Multi-Core mode is on, it will calculate GEMM with parfor otherwise it will calculate by cellfun locally.
%   TODO:
%       Need to improve GEMM performance here.

    res_cell = DepthwiseGEMM(ker_mat,im_mat,im_d,multiplier,out_size,window_shape);
%   Reshape result cell into tensor format to match the output shape
    res = fi(zeros([out_size,im_d*multiplier]),t,f);

%   TODO: improve reshape operation.
    for i=1:im_d
        ch_res = permute(reshape(res_cell{i}',[fliplr(out_size),multiplier]),[2,1,3]);
        res(:,:,1+(i-1)*multiplier:i*multiplier)=ch_res;
    end
end

function res_cell = DepthwiseGEMM(ker_mat,im_mat,im_d,multiplier,out_size,window_shape)
    num_core = GetCurrentCore();
    cell_FLAG = 0;
    if  num_core>0 && cell_FLAG
        ker_cell = mat2cell(ker_mat,ones(1,im_d)*multiplier,prod(window_shape))';
        im_cell = mat2cell(im_mat,prod(window_shape),prod(out_size)*ones(1,im_d));
        
        res_cell = cell(1,im_d);
        parfor i=1:im_d
            res_cell{i}=ker_cell{i}*im_cell{i};
        end
    elseif num_core>0 && ~cell_FLAG
        res_cell = cell(1,im_d);
        im_mat_sp = size(im_mat);
        im_blk_len = im_mat_sp(2)/im_d;
        parfor i=1:im_d
            ker_blk = ker_mat((i-1)*multiplier+1:i*multiplier,:);
            im_blk = im_mat(:,(i-1)*im_blk_len+1:i*im_blk_len);
            res_cell{i}= ker_blk*im_blk;
        end
    else
        ker_cell = mat2cell(ker_mat,ones(1,im_d)*multiplier,prod(window_shape))';
        im_cell = mat2cell(im_mat,prod(window_shape),prod(out_size)*ones(1,im_d));
        res_cell = cellfun(@fimtimes,ker_cell,im_cell,'UniformOutput',false);
    end

%         ker_cell = mat2cell(ker_mat,ones(1,im_d)*multiplier,prod(window_shape))';
%         im_cell = mat2cell(im_mat,prod(window_shape),prod(out_size)*ones(1,im_d));
%         for i=1:im_d
%             par_res(i)= parfeval(@fimtimes,1,ker_cell{i},im_cell{i});
%         end
%         res_cell = cell(1,im_d);
%         for i=1:im_d
%             [Idx,value]=fetchNext(par_res);
%             res_cell{Idx}=value;
%         end
end