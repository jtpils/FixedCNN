function res = ReLU(im,t,f)
    res = fi(zeros(size(im)),t,f);
    res(im>0)=im(im>0);
end
    