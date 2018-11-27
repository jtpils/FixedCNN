% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: Pooling function for 2d pooling of input tensor

function res = MultiCoreGEMM(mat_a,mat_b)
    [ah,aw] = size(mat_a);
    [bh,bw] = size(mat_b);
    if aw~=bh
        error("Matrix Inner Dimension For Multiplication Must Match.");
    end
    
    n = GetCurrentCore();
    if n>0
        if ah<n && bw>n
            tmp_a = mat_b';
            mat_b = mat_a';
            mat_a = tmp_a;
        
            ablk_h = ceil(bw/n);
            ablk = mat2cell(mat_a,[ablk_h*ones(1,n-1),bw-ablk_h*(n-1)],[aw]);
            spmd
                tmp = ablk{labindex}*mat_b;
            end
            res = [tmp{1};tmp{2};tmp{3};tmp{4};tmp{5};tmp{6}]';
        elseif ah>n
            ablk_h = ceil(ah/n);
            ablk = mat2cell(mat_a,[ablk_h*ones(1,n-1),ah-ablk_h*(n-1)],[aw]);
            spmd
                tmp = ablk{labindex}*mat_b;
            end
            res = [tmp{1};tmp{2};tmp{3};tmp{4};tmp{5};tmp{6}];
        elseif ah<n && bw<n
            res = mat_a*mat_b;
        end
    else
        res = mat_a*mat_b;
        warning_info =['Multi-Core Mode GEMM is OFF, maybe something wrong with PCT config. ' ...
              'GEMM core will continue to run on Single-Core Mode without acceleration benefit.'];
        warning(warning_info);
    end
end