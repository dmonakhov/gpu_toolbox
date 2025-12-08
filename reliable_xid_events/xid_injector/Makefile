# XID Injector Makefile

NVCC ?= nvcc
NVCCFLAGS := -O2

all: xid_inject

xid_inject: xid_inject.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f xid_inject

# Quick test - trigger XID 31
test31: xid_inject
	@echo "=== Triggering XID 31 (page fault) ==="
	@echo "Watch dmesg in another terminal: sudo dmesg -w | grep -i xid"
	@echo ""
	./xid_inject 31

# Test with eBPF tracer (if built)
test-with-tracer: xid_inject
	@echo "Run in terminal 1: sudo ../xid_tracer/xid_tracer"
	@echo "Run in terminal 2: ./xid_inject 31"

.PHONY: all clean test31 test-with-tracer
