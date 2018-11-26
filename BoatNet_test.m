clc;
var_list = {'t','f','wordlen','fraclen','roundm','img','cnn_par','net','layer_res'};
clear(var_list{:});
wordlen =64;
fraclen =32;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

img = load('test_img\mat_img\img.mat');
cnn_par = load('model.mat');
net = cnn_par.net;
img = fi(img.img1,t,f);

resfile = 'recog_res.txt';
fileID = fopen(resfile,'w');
names = {"ship", "land", "sea"};

profile on
for i=1:10
    layer_res = BoatNet(img,net,t,f,roundm);
    [val,label] = max(layer_res.layers{12}.maps,[],3);
    fprintf(fileID,'%s\n',names{label});
end
profile viewer
fclose(fileID);