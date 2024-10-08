MPI_HOME ?= /opt/amazon/openmpi
NCCL_HOME ?= /usr/local/cuda/targets/x86_64-linux
CUDA_HOME ?= /usr/local/cuda

NVCC ?= nvcc
NVCUFLAGS +=  -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi

NVCUFLAGS +=  -I$(CUDA_HOME)/include
NVLDFLAGS += -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64

NVCUFLAGS += -DMPI_SUPPORT -I $(NCCL_HOME)/include
NVLDFLAGS +=  -L$(NCCL_HOME)/lib -lnccl

BIN_FILES_LIST := nccl_comm_test

build: ${BIN_FILES_LIST}
clean:
	rm -rf ${BIN_FILES_LIST}

%: %.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	$(NVCC) -o $@ $(NVCUFLAGS)  $< $(NVLDFLAGS)
