% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: Pooling function for 2d pooling of input tensor

function res = Pooling(im,t,f,window_shape,pool_type,stride,pad_method)
    poolreg ={'MAX','AVG'};
    poolregfunc = {'max','mean'};
    func_reg = containers.Map(poolreg,poolregfunc);
    switch nargin
        case 4
            res = PoolingByType(im,t,f,window_shape,[2,2],'SAME',@max);
        case 7
            try
                pool_func = func_reg(pool_type);
                res = PoolingByType(im,t,f,window_shape,stride,pad_method,str2func(pool_func));
            catch error_info
                disp(error_info);
                error("Unknown Pooling Type");
            end
        otherwise
            error("Input Parameters Not Match");
    end
end

function res = PoolingByType(im,t,f,window_shape,stride,pad_method,pool_func)
    [im_h,im_w,im_d] = size(im);
    channel_size = [im_h,im_w];
    
    [im,out_size,channel_size] = PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,pad_method);
    
    res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func);
end

function res = PoolingTensor(im,im_d,channel_size,out_size,window_shape,stride,pool_func)
    
% %   �˴������⣬�������ڳ�����ȵ�ͼ�񣬴�����
% %   ���㻬�����ڶ�Ӧ��im2col��ͼ��Ԫ�صı�ţ��ȼ����ţ�����ȡ���ݴ�����������ٽ��о���
% %   �˷��Ǽ��ټ���ʱ��ĺ��ģ��ڱ�Ҫ��ʵ�ʼ��㷢��֮ǰ����Ҫ����ȥ����fi����͸ı�fi������״��
% %   ��Ϊfi����Ĵ����͸ı�λ�÷ǳ���ʱ�������ܶ�ʹ�ñ��������ʵ��Ԫ�ء�

    tmp1 = GetElemPos(im_d,channel_size,out_size,window_shape,stride);

    res = permute(reshape(pool_func(im(tmp1)),[fliplr(out_size),im_d]),[2,1,3]);
end