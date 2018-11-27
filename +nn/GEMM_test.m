wordlen =64;
fraclen =32;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

ah = 3000;
aw = 1000;
bw = 2;

a = randi(128,ah,aw);
b = randi(128,aw,bw);

tic

t0 = toc;
res1 = MultiCoreGEMM(a,b);
t1 = toc;

res2 = a*b;
t2 = toc;

time1 = t1-t0;
time2 = t2-t1;
fprintf('MultiCore completed time is %f s\n',time1);
fprintf('Common completed time is %f s\n',time2)

sp_delt = sum(abs(size(res1)-[ah,bw]));
delt = sum(sum(abs(res1-res2)));

disp(sp_delt);
disp(delt);