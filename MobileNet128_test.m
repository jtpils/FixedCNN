% Author: Zhao Mingxin
% Date:   2018/12/09
% Description: 

wordlen =20;
fraclen =10;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);

im = load('cat_test.mat');
inputs = (double(im.cat_im)-128)*(1/256);

net_par = getJSONParams('mobilenet_v1_1.0_128_quant.json');

nn.TurnOnMultiCore();
res = runNetWithJsonParams(net_par,inputs,t,f);

[~,label] = max(res);
fprintf(2,'Output Label:  %d\n',label);

function res = getJSONParams(path)
    fprintf(2,'Loading parameters from json file...\n');
    res = jsondecode(fileread(path));
    fprintf(2,'Parameters loaded.\n');
end