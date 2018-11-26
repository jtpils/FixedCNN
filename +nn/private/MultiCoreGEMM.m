function res = MultiCoreGEMM(mat_a,mat_b)
    p = gcp('nocreate'); 
    if isempty(p)
        n = 0;
    else
        n = p.NumWorkers;
    end
    ker_h = ceil(k_out/n);    
    ker_block = mat2cell(mat_a,[ker_h*ones(1,n-1),k_out-ker_h*(n-1)],[k_in]);
    spmd
        tmp = ker_block{labindex}*mat_b;
    end
    res = [tmp{1};tmp{2};tmp{3};tmp{4};tmp{5};tmp{6}];
end