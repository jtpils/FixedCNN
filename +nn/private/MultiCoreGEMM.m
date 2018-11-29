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
    shape = [ah,aw,bw];
    n = GetCurrentCore();
    if n>0
        [cal_mode,row_per_wk] = TaskScheduler(n,ah,aw,bw);
        switch cal_mode
            case 'A_ROW'
                res = RowBlockMM(mat_a,mat_b,row_per_wk,shape);
            case 'B_COL'
                res = ColBlockMM(mat_a,mat_b,row_per_wk,shape);
            case 'A_B_BLK'
                res = BlockBlockMM(mat_a,mat_b,row_per_wk,shape);
            case 'SingleCore'
                res = mat_a*mat_b;
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

% NOTE: Cannon Algorithm is the well-known GEMM parallel method while MATLAB does't support 
% finely manipulate matrix between workers so I just adopt simple rules to schedule tasks.

% TODO: deal with aw>>ah condition

function  [cal_mode,row_per_wk] = TaskScheduler(n,ah,aw,bw)
    SplitFunc = @(row,n_wk) floor(row/n_wk)*ones(1,n_wk)+...
        [ones(1,mod(row,n_wk)),zeros(1,n_wk-mod(row,n_wk))];

    sp = [ah,bw];
    ubl_ft =  ceil(sp/n)./(sp/n)-1;
    
    LargeThresh = 128*128*128;
    alpha = 100;
    if LargeThresh - ah*aw*bw>0
        row_per_wk = zeros(1,n);
        cal_mode = 'SingleCore';
    elseif aw>alpha*ah && aw>alpha*bw
        row_per_wk = SplitFunc(aw,n);
        cal_mode = 'A_B_BLK';
    else
        if ubl_ft(1)<ubl_ft(2)
            row_per_wk = SplitFunc(ah,n);
            cal_mode = 'A_ROW';
        elseif ubl_ft(1)>=ubl_ft(2)
            row_per_wk = SplitFunc(bw,n);
            cal_mode = 'B_COL';
        else
            row_per_wk = zeros(1,n);
            cal_mode = 'Unknown';
        end
    end
end

% shape=[ah,aw,bw]
function res = RowBlockMM(mat_a,mat_b,row_per_wk,shape)
    a_blk = mat2cell(mat_a,row_per_wk,shape(2));
    spmd
        tmp_res = a_blk{labindex}*mat_b;
    end
    res = repmat(mat_a(1),[shape(1),shape(3)]);
    rowStart = 1;
    for i = 1:length(row_per_wk~=0)
        res(rowStart:rowStart+row_per_wk(i)-1,:)=tmp_res{i};
        rowStart = rowStart + row_per_wk(i);
    end
end

function res = ColBlockMM(mat_a,mat_b,row_per_wk,shape)
    b_blk = mat2cell(mat_b,shape(2),row_per_wk);
    spmd
        tmp_res = mat_a*b_blk{labindex};
    end
    res = repmat(mat_a(1),[shape(1),shape(3)]);
    colStart = 1;
    for i = 1:length(row_per_wk~=0)
        res(:,colStart:colStart+row_per_wk(i)-1)=tmp_res{i};
        colStart = colStart + row_per_wk(i);
    end
end

function res = BlockBlockMM(mat_a,mat_b,row_per_wk,shape)
    a_cell=mat2cell(mat_a,shape(1),row_per_wk);
    b_cell=mat2cell(mat_b,row_per_wk,shape(3));
    spmd
        res_tmp = a_cell{labindex}*b_cell{labindex};
    end
    res = repmat(mat_a(1),[shape(1),shape(3)]);
    for i=1:length(row_per_wk~=0)
        res=res_tmp{i}+res;
    end
end