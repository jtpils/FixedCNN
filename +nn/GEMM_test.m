wordlen=16;
fraclen =8;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

numCores = GetCurrentCore();

small_mat = {[20,50,50],[12,1000,20],[50,50,10],[20,120,100],[100,10,200]};
med_mat = {[195,512,512],[23,12000,45],[512,512,50],[30,1024,500],[1000,60,1000]};
large_mat = {[512,512,1024],[48,15000,64],[800,800,80],[80,2048,800],[2000,80,2000]};

mat_test = {small_mat,med_mat,large_mat};
matrix_type = {'SMALL','MEDIATE','LARGE'};
tic
for scale = 1:3
    info = strcat(matrix_type{scale},' Matrix GEMM is begining...\n');
    fprintf(2,info);
    shape_tb = mat_test{scale};
    for i=1:length(shape_tb)
        rsp = shape_tb{i};
        ah = rsp(1);
        aw = rsp(2);
        bw = rsp(3);
        a = fi(randi(128,ah,aw),t,f);
        b = fi(randi(128,aw,bw),t,f);
        fprintf(2,'GEMM TEST START with shape:[%d,%d,%d]...\n',ah,aw,bw);

        t0 = toc;
        res1 = MultiCoreGEMM(a,b);
        t1 = toc;
        fprintf('MultiCoreGEMM completed time is %f s\n',t1-t0);

        t2 = toc;
        res2 = a*b;
        t3 = toc;
        fprintf('Default GEMM completed time is %f s\n',t3-t2);

        t4 = toc;
        res3 = COLsplGEMM(a,b);
        t5 = toc;
        fprintf('ColsplGEMM completed time is %f s\n',t5-t4);
    end
end

function res3 = COLsplGEMM(a,b)
    [ah,aw]=size(a);
    [bh,bw]=size(b);
    if aw~=bh
        error('Dimension Doesn''t Match.');
    end
    n_wk = GetCurrentCore();
    row = aw;
    Colsplit = floor(row/n_wk)*ones(1,n_wk)+...
         [ones(1,mod(row,n_wk)),zeros(1,n_wk-mod(row,n_wk))];
    a_cell=mat2cell(a,ah,Colsplit);
    b_cell=mat2cell(b,Colsplit,bw);
    spmd
        res_tmp = a_cell{labindex}*b_cell{labindex};
    end
    res3 = repmat(a(1),[ah,bw]);
    for i=1:length(Colsplit~=0)
        res3=res_tmp{i}+res3;
    end
end