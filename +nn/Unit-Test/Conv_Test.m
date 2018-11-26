addpath
im_ker = cat(3,ker,ker1,ker3,ker1,ker3,ker4);
in_ker = fi(reshape(im_ker,[3,3,3,2]),t,f);
stride = [1,1];
padding_method = 'VALID';
pp=DepthwiseConv2d(fi(dog,t,f),in_ker,t,f,stride,padding_method);