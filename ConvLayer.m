function o_map = ConvLayer(im,ker,t,f)
    [im_h,im_w,im_d]=size(im);
    [k_h,k_w,k_in,k_out]=size(ker);
    
    stride=1;
    out_h = ceil(im_h+2-k_h)/stride+1;
    out_w = out_h;
    
    o_map = fi(zeros(out_h,out_w,k_out),t,f);
    for i=[1:1:im_d]
        o_map = o_map + ConvOneMap(im(:,:,i),ker(:,:,i,:),t,f);
    end
end