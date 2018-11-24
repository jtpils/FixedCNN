clc;
var_list = {'t','f','wordlen','fraclen','roundm','img','cnn_par','net','layer_res'};
clear(var_list{:});
wordlen =16;
fraclen = 8;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

img = load('test_img\mat_img\img.mat');
cnn_par = load('model.mat');
net = cnn_par.net;
profile on
for i=1:10
    layer_res = FixConvNet(img.img1,net,t,f,roundm);
end
profile viewer