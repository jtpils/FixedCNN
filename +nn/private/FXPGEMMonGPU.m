function res = FXPGEMMonGPU(mat_a,mat_b)
    if ~isfi(mat_a) || ~isfi(mat_b)
        error('GPU Fixed point GEMM only support fi object for now.');
    end
    blk_size = 16;
    
    t = mat_a.numerictype;
    f = mat_a.fimath;
    FracLen = t.FractionLength;
    WordLen = t.WordLength;
    
    if 2*WordLen<32
        over_bound = 2^(f.ProductWordLength-1);
    else
        over_bound = 2^(23-1);
    end
    
    [ah,aw]=size(mat_a);
    [bh,bw]=size(mat_b);
    
    if aw~=bh
        error('Inner Dimension Must Match.');
    end
    
    ah_n = ceil(ah/blk_size);
    aw_n = ceil(aw/blk_size);
    bh_n = ceil(bh/blk_size);
    bw_n = ceil(bw/blk_size);
    
    a_pad = single(zeros(ah_n*blk_size,aw_n*blk_size));
    b_pad = single(zeros(bh_n*blk_size,bw_n*blk_size));
    
    a_int = mat_a.data*(2^mat_a.FractionLength);
    b_int = mat_b.data*(2^mat_b.FractionLength);
    
    a_pad(1:ah,1:aw)=single(a_int);
    b_pad(1:bh,1:bw)=single(b_int);
    
    gpu_kernel = parallel.gpu.CUDAKernel('+nn/private/matmul.ptx','+nn/private/matmul.cu');
    gpu_kernel.GridSize=[bw_n,ah_n,1];
    gpu_kernel.ThreadBlockSize=[blk_size,blk_size,1];
    
    mat_c = single(zeros(ah_n*blk_size,bw_n*blk_size));
    res_gpu = feval(gpu_kernel,a_pad,b_pad,ah_n*blk_size,aw_n*blk_size,bw_n*blk_size,over_bound,mat_c);
    res = gather(res_gpu);
    
    res = fi(res(1:ah,1:bw)/2^(2*FracLen),t,f);
end
    
    