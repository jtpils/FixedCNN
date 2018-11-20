randFlag=1;
if randFlag==1
    ker = randi([0,128],3,3);
end
csvwrite('kernel.csv',ker);

function impad = ZeroPadding(im,t,f)
    im_shape = size(im);
    impad = repmat(fi(0,t,f),im_shape+2);
    impad(2:end-1,2:end-1)=im;
end

function out_map = ConvOneChannel(im,ker,t,f)
    im_mat = im2col(ZeroPadding(im,t,f),size(ker),'sliding');
    ker_mat = reshape(ker,[1,numel(ker)]);
    out_map = reshape(ker_mat*im_mat,size(im));
end
