clc;clear all;
f = fimath('CastBeforeSum', 0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength', 16, ...
'ProductFractionLength', 8, 'SumWordLength', 16, 'SumFractionLength', 8);
t = numerictype('WordLength', 16, 'FractionLength', 8);
roundm = 'floor';

img = load('test_img\mat_img\img.mat');
cnn_par = load('model.mat');
net = cnn_par.net;
profile on
for i=1:20
    layer_res = FixConvNet(img.img1,net,t,f,roundm);
end
profile viewer