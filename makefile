
NVCC = nvcc
CUDAFLAGS = -lcuda -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -O3
LIBDIRS = -I.

search: Search.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) -o search Search.cu -lm 
clean:
	\rm *.o search
