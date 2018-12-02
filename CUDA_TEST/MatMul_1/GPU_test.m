wordlen =16;
fraclen =8;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

ah = 128;
aw = 256;
bw = 256;

mat_a = fi(single(rand(ah,aw)),t,f);
mat_b = fi(single(rand(aw,bw)),t,f);

tic
t1 = toc;
res_cpu = mat_a*mat_b;
t2 = toc;
fprintf('CPU GEMM time is %f s\n',t2-t1);

t3 = toc;
res_gpu = FXPGEMMonGPU(mat_a,mat_b);
t4 = toc;
fprintf('GPU GEMM time is %f s\n',t4-t3);

sum(sum(abs(single(res_cpu-res_gpu))))