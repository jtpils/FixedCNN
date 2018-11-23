clc;clear all;
f = fimath('CastBeforeSum', 0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength', 32, ...
'ProductFractionLength',16, 'SumWordLength', 32, 'SumFractionLength', 16);
t = numerictype('WordLength', 32, 'FractionLength',16);
roundm = 'floor';

img = load('test_img\mat_img\img.mat');
cnn_par = load('model.mat');
net = cnn_par.net;
profile on
for i=1:50
    layer_res = FixConvNet(img.img1,net,t,f,roundm);
end
profile viewer