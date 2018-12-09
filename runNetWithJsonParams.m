% Author: Zhao Mingxin
% Date:   2018/12/09
% Description: 

function res = runNetWithJsonParams(par,inputs,t,f)
    op_parse = par.subgraphs.operators;
    tensor_list = par.subgraphs.tensors;
    tensor_buffer = par.buffers;
    
    for i=1:length(op_parse)
        if i==1
            net = inputs;
        end
        switch op_parse(i).opcode_index
            case 1
                par_n = op_parse(i).inputs;
                conv_op = op_parse(i).builtin_options;
                stride = [conv_op.stride_h,conv_op.stride_w];
                padding = conv_op.padding;
                
                weight_n = par_n(2)+1;
                bias_n = par_n(3)+1;
                conv_w = tensor_buffer{tensor_list(weight_n).buffer+1,1}.data;
                bias_w = tensor_buffer{tensor_list(bias_n).buffer+1,1}.data;
                
                conv_w_tf = getTFStyleParams(conv_w,tensor_list(weight_n).shape,'Conv2d');
                bias_w_tf = getBias(bias_w);
                conv_w = dequantParams(conv_w_tf,'Weight',tensor_list(weight_n).quantization);
                bias_w = dequantParams(bias_w_tf,'Bias',tensor_list(bias_n).quantization);
                
                conv_fi = fi(conv_w,t,f);
                bias_fi = fi(bias_w,t,f);
                net = nn.Conv2d(net,conv_fi,t,f,stride,padding);
                net = nn.AddBias(net,bias_fi,t,f);
                net = nn.ReLU(net);
            case 2
                par_n = op_parse(i).inputs;
                conv_op = op_parse(i).builtin_options;
                stride = [conv_op.stride_h,conv_op.stride_w];
                padding = conv_op.padding;
                
                weight_n = par_n(2)+1;
                bias_n = par_n(3)+1;
                conv_w = tensor_buffer{tensor_list(weight_n).buffer+1,1}.data;
                bias_w = tensor_buffer{tensor_list(bias_n).buffer+1,1}.data;
                
                conv_w_tf = getTFStyleParams(conv_w,tensor_list(weight_n).shape,'DepthwiseConv2d');
                bias_w_tf = getBias(bias_w);
                conv_w = dequantParams(conv_w_tf,'Weight',tensor_list(weight_n).quantization);
                bias_w = dequantParams(bias_w_tf,'Bias',tensor_list(bias_n).quantization);
                
                conv_fi = fi(conv_w,t,f);
                bias_fi = fi(bias_w,t,f);
                net = nn.DepthwiseConv2d(net,conv_fi,t,f,stride,padding);
                net = nn.AddBias(net,bias_fi,t,f);
                net = nn.ReLU(net);
            case 3
                fprintf(2,'Softmax Layer Detected.\n');
            case 4
                shape_op = op_parse(i).builtin_options;
                new_shape = shape_op.new_shape;
                net = reshape(net,new_shape);
            case 0
                pool_op = op_parse(i).builtin_options;
                padding = pool_op.padding;
                stride = [pool_op.stride_h,pool_op.stride_w];
                window_shape = [pool_op.filter_height,pool_op.filter_width];
                net = nn.Pooling(net,t,f,window_shape,'AVG',stride,padding);
            otherwise
                warning('Unknown OP type detected.');
        end
    end
    res = net;
end

% The first argument par is a 1-dim array which is little-endian format according to fbs
% definition. The second arg shape typically is a [N,C,H,W] array. This function aims to 
% transform flatc_buffer pre-defined parameters to TensorFlow-style
% parameters so that we can use these params to perform inference using
% FixedCNN simulation library.
function res = getTFStyleParams(par,shape,layer_type)
    switch layer_type
        case 'Conv2d'
            tmp = permute(reshape(par,fliplr(shape')),[4,3,2,1]);
            res = permute(tmp,[2,3,4,1]);
        case 'DepthwiseConv2d'
            tmp = permute(reshape(par,fliplr(shape')),[4,3,2,1]);
            res = permute(tmp,[2,3,4,1]);
        otherwise
            error('Unknown Layer Type.');
    end
end

% This function used to dequantize tflite fixed-point params.
function res = dequantParams(par,par_type,quant)
    switch par_type
        case 'Weight'
            zp = quant.zero_point;
            scale = quant.scale;
            res = scale*(double(par)-zp);
        case 'Bias'
            zp = quant.zero_point;
            scale = quant.scale;
            res = scale*(double(par)-zp);
        otherwise
            error('Unknown Params Type');
    end
end

function res = getBias(ubias)
    int8bias = reshape(ubias,4,[])';
    [h,~]=size(int8bias);
    bias_cell = mat2cell(int8bias,ones(1,h),4);
    double_bias = cellfun(@(x) double(typecast(uint8(x),'int32')),bias_cell);
    res = double_bias';
end