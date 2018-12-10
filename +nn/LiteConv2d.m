function res = LiteConv2d(im,ker,z_im,z_ker,z_res,s1,s2,s3,ConvType,stride,padding,bias)
    wordlen = 32;
    fraclen = 0;
    
    f = fimath('CastBeforeSum',0, 'OverflowMode', 'Saturate', 'RoundMode', 'floor', ... 
    'ProductMode', 'SpecifyPrecision', 'SumMode', 'SpecifyPrecision', 'ProductWordLength',wordlen, ...
    'ProductFractionLength',fraclen, 'SumWordLength', wordlen, 'SumFractionLength', fraclen);
    t = numerictype('WordLength', wordlen, 'FractionLength',fraclen);
    
    im_int = fi(im,t,f)-fi(z_im,t,f);
    ker_int = fi(ker,t,f)-fi(z_ker,t,f);
    
    % Calculate by type
    switch ConvType
        case 'Conv2d'
            conv_res = nn.Conv2d(im_int,ker_int,t,f,stride,padding);
        case 'DepthwiseConv2d'
            conv_res = nn.DepthwiseConv2d(im_int,ker_int,t,f,stride,padding);
        otherwise
            error('Unknown ConvType Detected.');
    end
    conv_res = nn.AddBias(conv_res,fi(bias,t,f),t,f);
    
    % OutputStage
    [mul,n] = getShiftBits(s1,s2,s3,16);
    res = bitshift(conv_res*mul,-n)+ fi(z_res,t,f);
    res = fi(res,0,8,0);
end


function [mul,n] = getShiftBits(s1,s2,s3,base)
    M = s1*s2/s3;
    n0 = 0;
    while M<0.5
        M = M*2;
        n0 = n0+1;
    end
    mul = floor(M*2^(base-1));
    n = n0-1+base;
end