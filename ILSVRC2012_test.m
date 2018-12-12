% Author: Zhao Mingxin
% Date:   2018/12/11
% Description: Run tflite fixed point CNN using JSON parameters.
%{
    JSON file should be converted from .tflite file using flatc interpreter. 
    After you extract network parameters, you can call this function to run
    a fixed point CNN. More details please refer to TensorFlow API and
    Google 8-bit quantization paper:
    
    If you have any issues about this code, please feedback.
%}

wordlen =32;
fraclen =0;
f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'nearest', ... 
'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',wordlen, ...
'ProductFractionLength',fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);

lbfile = load('validation_lbs.mat');
val = lbfile.val_lbs;

net_par = getJSONParams('mobilenet_v1_1.0_128_quant.json');

val_path = '/home/zhaomingxin/Datasets/ILSVRC2012/val/ILSVRC2012_img_val/';

resfile = 'ILSVC2012_VAL_RES.txt';
fileID = fopen(resfile,'w');

totalNum = 0;
correctNum = 0;
tic;
t0 = toc;
for i=1:2000
    imname = char(strcat(val_path,val.FileName(i)));
    imlb = val.Label(i);
    try
        img = imread(imname);
        input = impreprocess(img);
        [h,w,d] = size(input);
        if d~=3
            fprintf('Image Depth Not 3 in Loop: %d\n',i);
            continue
        end
    catch
        fprintf('Something wrong in this loop: %d\n',i);
        continue
    end
    totalNum = totalNum + 1;
    res = runLiteWithJsonParams(net_par,input,t,f);
    pred = getTopKPred(res,5);
    if ismember(imlb+2,pred)
        correctNum = correctNum+1;
    end
    fprintf(fileID,'Image ID: %5d,  Pred: %5d,   Total: %5d,  Correct: %5d \n',i,ismember(imlb+2,pred),totalNum,correctNum);
end
t1 = toc;
disp(t1-t0);
fclose(fileID);

fprintf(2,'Validation Completed.\n');
fprintf('Accuracy is %2.2f %% \n',correctNum/totalNum*100.0);

function res = impreprocess(img)
    [h,w,~]=size(img);
    if h>w
        scale = floor(128*[h/w,1]);
        tiny_im = imresize(img,scale);
%         pos = randi(max(scale-128));
        pos = max(floor((scale-128)/2));
        if pos==0
            pos = 1;
        end
        res = tiny_im(pos:pos+127,:,:);
    elseif h<w
        scale = floor(128*[1,w/h]);
        tiny_im = imresize(img,scale);
%         pos = randi(max(scale-128));
        pos = max(floor((scale-128)/2));
        if pos==0
            pos = 1;
        end
        res = tiny_im(:,pos:pos+127,:);
    else
        res = imresize(img,[128,128]);
    end
end

function res = getTopKPred(pred,k)
    [~,idx] = sort(pred,'descend');
    tmp = idx(1:k);
    res = tmp;
end

function res = getJSONParams(path)
    fprintf(2,'Loading parameters from JSON file...\n');
    res = jsondecode(fileread(path));
    fprintf(2,'Parameters Loaded.\n');
end