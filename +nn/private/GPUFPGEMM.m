% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below

%{
    I will implement fixed point GEMM on GPU by truncating intermediate
    results with fixed point data format in the future.
%}

function res = GPUFPGEMM(mat_a,mat_b)
    if ~isfi(mat_a) || ~isfi(mat_b)
        error('GPU Fixed point GEMM only support fi object for now.');
    end
    
    t = mat_a.numerictype;
    f = mat_a.fimath;
    FracLen = t.FractionLength;
    
    % Using single instead of double because GPU is more efficient for floating
    % point operations.
    
    % TODO:
    % change default gpu gemm to customed GPUGEMM function.
    a = gpuArray(single(mat_a.int));
    b = gpuArray(single(mat_b.int));
    res_gpu = a*b;
%    res_gpu = FXPGEMMonGPU(a,b,t,f);
    
    res_cpu = gather(res_gpu);
    res = fi(res_cpu/2^FracLen,t,f);
end

function res = FXPGEMMonGPU(a,b,t,f)

end