wordlen =64;
fraclen =32;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

ah = 60;
aw = 10000;
bw = 50;

a = fi(randi(128,ah,aw),t,f);
b = fi(randi(128,aw,bw),t,f);

tic

t0 = toc;
profile on
res1 = MultiCoreGEMM(a,b);
t1 = toc;

fprintf('MultiCore completed time is %f s\n',t1-t0);


t2 = toc;
res2 = a*b;
t3 = toc;
fprintf('Common completed time is %f s\n',t3-t2);
profile viewer

sp_delt = sum(abs(size(res1)-[ah,bw]));
delt = sum(sum(abs(res1-res2)));

disp(double([sp_delt,delt]));