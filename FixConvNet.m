function cnn_result = FixConvNet(im,cnn_net,t,f,rounding_method)

input_img.RoundingMethod = rounding_method;
cnn_result.layers{1}.maps = im;
cnn_layers = numel(cnn_net.layers);

for i = 1:cnn_layers
	net_layer = cnn_net.layers{i};
    map_layer = cnn_result.layers{i};
	switch net_layer.type
		case 'conv'
            net_layer.filters = fi(net_layer.weights{1},t,f);%Preprocess
            net_layer.bias = fi(net_layer.weights{2},t,f);
            net_layer.filters.RoundingMethod = rounding_method;
            net_layer.bias.RoundingMethod = rounding_method;
% 			cnn_result.layers{i+1}.maps = AddBias(ConvLayer(map_layer.maps, net_layer.filters,t,f),net_layer.bias,t,f);
            cnn_result.layers{i+1}.maps = AddBias(Conv2d(map_layer.maps, net_layer.filters,t,f,[1,1],'SAME'),net_layer.bias,t,f);
		case 'pool'
            poolstride = net_layer.stride*ones(1,2);
            method = upper(net_layer.method);
            if length(net_layer.pad)>1
                pad_method = 'SAME';
            else
                pad_method = 'VALID';
            end
			cnn_result.layers{i+1}.maps = Pooling(map_layer.maps,t,f,net_layer.pool ,method,poolstride,pad_method); % Pooling(im,t,f,poolsize,pool_type,poolstride,pad_method)
		case 'relu'
			cnn_result.layers{i+1}.maps = ReLU(map_layer.maps);
		case 'softmaxloss'
			disp('softmaxloss layer');
		case 'softmax'
			disp('softmax layer');
		otherwise
% 			error('Unknown layer type %s', net_layer.type);
            dstr = sprintf('Warning! Unknown layer type %s \n', net_layer.type);
            fprintf(dstr);
	end
end
end