% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: The core function for GEMM with MultiCore support.

function res = MultiCoreGEMM(mat_a,mat_b)
    [ah,aw] = size(mat_a);
    [bh,bw] = size(mat_b);
    if aw~=bh
        error("Matrix Inner Dimension For Multiplication Must Match.");
    end
    n = GetCurrentCore();
    if n>0
        [cal_mode,row_per_wk] = TaskScheduler(n,ah,bw);
        switch cal_mode
            case 'A_ROW'
                a_blk = mat2cell(mat_a,row_per_wk,aw);
                spmd
                    tmp_res = a_blk{labindex}*mat_b;
                end
                res = repmat(mat_a(1),[ah,bw]);
                rowStart = 1;
                for i = 1:length(row_per_wk~=0)
                    res(rowStart:rowStart+row_per_wk(i)-1,:)=tmp_res{i};
                    rowStart = rowStart + row_per_wk(i);
                end
            case 'B_COL'
                b_blk = mat2cell(mat_b',row_per_wk,bh);
                spmd
                    tmp_res = b_blk{labindex}*mat_a';
                end
                col_res = repmat(mat_a(1),[bw,ah]);
                rowStart = 1;
                for i = 1:length(row_per_wk~=0)
                    col_res(rowStart:rowStart+row_per_wk(i)-1,:)=tmp_res{i};
                    rowStart = rowStart + row_per_wk(i);
                end
                res = col_res';
            otherwise
                error('Unknown Computation Mode Detected.');
        end
    else
        res = mat_a*mat_b;
        warning_info =['Multi-Core Mode GEMM is OFF, maybe something wrong with PCT config. ' ...
              'GEMM core will continue to run on Single-Core Mode without acceleration benefit.'];
        warning(warning_info);
    end
end

function  [cal_mode,row_per_wk] = TaskScheduler(n,ah,bw)
    RowSplit = @(row,n_wk) floor(row/n_wk)*ones(1,n_wk)+...
            [ones(1,mod(row,n_wk)),zeros(1,n_wk-mod(row,n_wk))];
    if ah>n
        row_per_wk = RowSplit(ah,n);
        cal_mode = 'A_ROW';
    elseif bw>n
        row_per_wk = RowSplit(bw,n);
        cal_mode = 'B_COL';
    else
        row_per_wk = RowSplit(ah,n);
        cal_mode = 'A_ROW';
    end
end