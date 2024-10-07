MPI_HOME ?= /opt/amazon/openmpi
NCCL_HOME ?= /opt/nccl/build
NVCC ?= nvcc
NVCUFLAGS +=  -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
NVCUFLAGS += -DMPI_SUPPORT -I$(NCCL_HOME)/include
NVLDFLAGS +=  -L$(NCCL_HOME)/lib -lnccl

BIN_FILES_LIST := nccl_comm_test

build: ${BIN_FILES_LIST}
clean:
	rm -rf ${BIN_FILES_LIST}

%: %.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	$(NVCC) -o $@ $(NVCUFLAGS)  $< $(NVLDFLAGS)
