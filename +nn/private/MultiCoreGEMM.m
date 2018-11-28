% Author: Zhao Mingxin
% Date:   2018/11/27
% Description: The core function for GEMM with MultiCore support.
%{
    NOTE: Up to now, the GEMM function will give obviously acceleration for
    LARGE matrix-matrix multiplication but mediate and small scale matrix-matrix 
    multiplication will suffer from communication delay between different cores 
    so that the time becomes longer than ordinary MM inevitablely.
%}

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
           % TODO: Reduce reshape time
                rowStart = 1;
                for i = 1:length(row_per_wk~=0)
                    res(rowStart:rowStart+row_per_wk(i)-1,:)=tmp_res{i};
                    rowStart = rowStart + row_per_wk(i);
                end
            case 'B_COL'
                b_blk = mat2cell(mat_b,bh,row_per_wk);
                spmd
                    tmp_res = mat_a*b_blk{labindex};
                end
                res = repmat(mat_a(1),[ah,bw]);
           % TODO: Reduce reshape time
                colStart = 1;
                for i = 1:length(row_per_wk~=0)
                    res(:,colStart:colStart+row_per_wk(i)-1)=tmp_res{i};
                    colStart = colStart + row_per_wk(i);
                end
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

% TaskScheduler accepts [worker numbers n, mat_a and mat_b shape] and 
% arranges rows of matrix to different workers to perform MultiCoreGEMM. The rules mainly due to
% height of mat_a and width of mat_b. 

% NOTE: Cannon Algorithms is the well-known GEMM parallel method while MATLAB does't support 
% finely manipulate matrix between workers so I just adopt simple rules to schedule tasks.
function  [cal_mode,row_per_wk] = TaskScheduler(n,ah,bw)
    RowSplit = @(row,n_wk) floor(row/n_wk)*ones(1,n_wk)+...
            [ones(1,mod(row,n_wk)),zeros(1,n_wk-mod(row,n_wk))];
    if (ah>=bw && bw>=n) || (ah>=n && n>=bw) || (n>=ah && ah>=bw)
        row_per_wk = RowSplit(ah,n);
        cal_mode = 'A_ROW';
    elseif (bw>=ah && ah>=n) || (n>=bw && bw>=ah) || (bw>=n && n>= ah)
        row_per_wk = RowSplit(bw,n);
        cal_mode = 'B_COL';
    else
        row_per_wk = zeros(1,n);
        cal_mode = 'Unknown';
    end
end