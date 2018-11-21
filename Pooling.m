function res = Pooling(im,t,f,poolsize)
    [im_h,im_w,im_d]=size(im);
    b_h = poolsize(1);
    b_w = poolsize(2);
    block_rows = im_h/poolsize(1);
    block_cols = im_w/poolsize(2);
    im_block = mat2cell(im,b_h*ones(1,block_rows),b_w*ones(1,block_cols),[im_d]);
    tmp = cellfun(@(x) max(reshape(x,b_h*b_w,im_d)),im_block,'UniformOutput',false);
    tmp = cellfun(@(x) x.data,tmp,'UniformOutput',false);
    res_r = cell2mat(tmp);
    res = fi(res_r,t,f);
    res = reshape(res',3,[]);
    res = permute(reshape(res',16,16,3),[2,1,3]);
end