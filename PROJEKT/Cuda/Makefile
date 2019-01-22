CFLAGS += -x cu -arch=sm_35  -D__CUDA
CXX = nvcc

RSA_CUDA: RSA_CUDA.cpp
	$(CXX) $(CFLAGS) -o RSA_CUDA RSA_CUDA.cpp
