% Author: Zhao Mingxin
% Date:   2018/12/09
% Description: 

wordlen =16;
fraclen =8;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',wordlen, ...
'ProductFractionLength',fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);

labels = load('labels_1001.mat');
lbs = labels.category;
samples = load('samples.mat');

samp = samples.samples(7);

img = samp.images{1,1};

% img = imresize(img7,[128,128]);

% inputs = (double(img)-128)*(2/256);

load_FLAG = 1;
if load_FLAG
    net_par = getJSONParams('mobilenet_v1_1.0_128_quant.json');
end

device.CPU.callMultiCore();
tic

t_Start = toc;
% res = runNetWithJsonParams(net_par,inputs,t,f);
res = runLiteWithJsonParams(net_par,img,t,f);
t_End = toc;

getTopKPred(res,5,lbs);

fprintf(2,'True Category: %s \n',string(samp.category));

fprintf('MobileNet-128-1.0 completed in %fs\n',t_End-t_Start);

function res = getJSONParams(path)
    fprintf(2,'Loading parameters from JSON file...\n');
    res = jsondecode(fileread(path));
    fprintf(2,'Parameters Loaded.\n');
end

function res = getTopKPred(pred,k,lbs)
    [~,idx] = sort(pred,'descend');
    tmp = idx(1:k);
    fprintf('Top-%d Labels: \n',k);
    for i = 1:k
        fprintf('%d  ',tmp(i));
    end
    fprintf('\n');
    fprintf('Classification Results: \n');
    for i = 1:k
        fprintf('\t %s \n',string(lbs(tmp(i),:)));
    end
    res = tmp;
end