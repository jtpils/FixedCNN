__global__ void DepthwiseGEMM(const float *A,const float *B,const int Aheight,const int Awidth,const int Bwidth, const float over_bound,float *C)
{
	float Cvalue = 0;
	float prod_tmp;
	int Bheight = Awidth;
	
	int chn = blockIdx.z;

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	
	for (int e = 0; e < Awidth; ++e){
		prod_tmp = A[chn * Aheight * Awidth + Aheight * e + row]*B[chn*Bheight*Bwidth + col * Bheight+ e];
		if(prod_tmp>over_bound)
			prod_tmp=over_bound;
		Cvalue+=prod_tmp;
		if(Cvalue>over_bound)
			Cvalue=over_bound;
	}	
	C[chn*Aheight*Bwidth + Aheight*col + row] = Cvalue;
}