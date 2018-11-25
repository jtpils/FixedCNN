% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: as below

wordlen =16;
fraclen =8;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',2*wordlen, ...
'ProductFractionLength',2*fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
roundm = 'floor';

% MobileNet Begin
profile on

% Layer 1
net = Conv2d(fi(randi(64,224,224,3),t,f),fi(randi(64,3,3,3,32),t,f),t,f,[2,2],'SAME');
net = AddBias(net,fi(randi(64,1,32)),t,f);
net = ReLU(net);
% Layer 2
net = DepthwiseConv2d(net,fi(randi(64,3,3,32,1)),t,f,[1,1],'SAME');
 
% Layer 3
net = Conv2d(net,fi(randi(64,1,1,32,64),t,f),t,f,[1,1],'SAME');

% Layer 4
net = DepthwiseConv2d(net,fi(randi(64,3,3,64,1)),t,f,[2,2],'SAME');

% Layer 5
net = Conv2d(net,fi(randi(64,1,1,64,128),t,f),t,f,[1,1],'SAME');

% Layer 6
net = DepthwiseConv2d(net,fi(randi(64,3,3,128,1)),t,f,[1,1],'SAME');

% Layer 7
net = Conv2d(net,fi(randi(64,1,1,128,128),t,f),t,f,[1,1],'SAME');

% Layer 8
net = DepthwiseConv2d(net,fi(randi(64,3,3,128,1)),t,f,[2,2],'SAME');

% Layer 9
net = Conv2d(net,fi(randi(64,1,1,128,256),t,f),t,f,[1,1],'SAME');

% Layer 10
net = DepthwiseConv2d(net,fi(randi(64,3,3,256,1)),t,f,[1,1],'SAME');
 
% Layer 11
net = Conv2d(net,fi(randi(64,1,1,256,256),t,f),t,f,[1,1],'SAME');

% Layer 12
net = DepthwiseConv2d(net,fi(randi(64,3,3,256,1)),t,f,[2,2],'SAME');

% Layer 13
net = Conv2d(net,fi(randi(64,1,1,256,512),t,f),t,f,[1,1],'SAME');

% Layer 14~23
for i=1:5
    net = DepthwiseConv2d(net,fi(randi(64,3,3,512,1)),t,f,[1,1],'SAME');
    net = Conv2d(net,fi(randi(64,1,1,512,512),t,f),t,f,[1,1],'SAME');
end

% Layer 24
net = DepthwiseConv2d(net,fi(randi(64,3,3,512,1)),t,f,[2,2],'SAME');

% Layer 25
net = Conv2d(net,fi(randi(64,1,1,512,1024),t,f),t,f,[1,1],'SAME');

% Layer 26
net = DepthwiseConv2d(net,fi(randi(64,3,3,1024,1)),t,f,[1,1],'SAME');

% Layer 27
net = Conv2d(net,fi(randi(64,1,1,1024,1024),t,f),t,f,[1,1],'SAME');

% Layer 28
net = Pooling(net,t,f,[7,7],'AVG',[1,1],'VALID');

% Layer Ouput
net = reshape(net,[],1024);

out = net*fi(randi(64,1024,1000),t,f);
% MobileNet End
profile viewer