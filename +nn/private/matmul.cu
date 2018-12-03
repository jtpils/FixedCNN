__global__ void MatMulKernel(const float *A,const float *B,const int Aheight,const int Awidth,const int Bwidth, const float over_bound,float *C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	float prod_tmp;
	int Bheight = Awidth;
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
/* 	for (int e = 0; e < Awidth; ++e)
		Cvalue += A[row * Awidth + e]*B[e * Bwidth + col];
	C[row * Cwidth + col] = Cvalue; */
	
/* 	for (int e = 0; e < Awidth; ++e){
		
		Cvalue += A[Aheight * e + row]*B[col * Bheight + e];
		
	}	
	C[Aheight*col + row] = Cvalue; */
	for (int e = 0; e < Awidth; ++e){
		prod_tmp = A[Aheight * e + row]*B[col * Bheight + e];
		if(prod_tmp>over_bound)
			prod_tmp=over_bound;
		Cvalue+=prod_tmp;
		if(Cvalue>over_bound)
			Cvalue=over_bound;
	}	
	C[Aheight*col + row] = Cvalue;
}