% Author: Zhao Mingxin
% Date:   2018/11/25
% Description: Padding function for Conv2d and Pooling, only supports SAME
%               and VALID mode with zeros padding.
%{ 
    Problems needed to solve 
    TODO: handle two problems for now
        problem 1: stride>channel_size
        problem 2: window_shape>channel_size
%}

function [res,out_size,new_channel_size] = PaddingByType(im,t,f,im_d,window_shape,channel_size,stride,padding_type)
    switch padding_type
        case 'SAME'
            out_size = ceil(channel_size./stride);
            pad_needed = (out_size-1).*stride + window_shape - channel_size;
            pad_top_left = floor(pad_needed/2);
 %           pad_bottom_right = pad_needed - pad_top_left;
            res = fi(zeros([channel_size+pad_needed,im_d]),t,f);           
            pad_end = pad_top_left + channel_size;
            res(1+pad_top_left(1):pad_end(1),1+pad_top_left(2):pad_end(2),:)=im;
        case 'VALID'
            out_size =ceil((channel_size - window_shape +1)./stride);
            pad_needed = [0,0];
            res = im;
        otherwise
            error('Unknown Padding Type');
    end
    new_channel_size = pad_needed + channel_size;
end