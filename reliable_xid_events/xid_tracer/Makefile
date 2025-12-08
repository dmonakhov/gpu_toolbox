# XID Tracer Makefile
#
# Prerequisites:
#   - Go 1.21+
#   - clang/llvm (for eBPF compilation)
#   - libbpf headers (optional, for full vmlinux.h generation)
#
# Build:
#   make build
#
# Run:
#   sudo ./xid_tracer

.PHONY: all generate build clean vmlinux

CLANG ?= clang
CFLAGS := -O2 -g -Wall -Werror -target bpf

all: build

# Generate Go bindings from eBPF C code
generate:
	go generate ./...

# Build the Go binary
build: generate
	go build -o xid_tracer .

# Generate full vmlinux.h from running kernel (requires bpftool and BTF-enabled kernel)
vmlinux:
	@if [ -f /sys/kernel/btf/vmlinux ]; then \
		bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h; \
		echo "Generated vmlinux.h from kernel BTF"; \
	else \
		echo "ERROR: /sys/kernel/btf/vmlinux not found. Is CONFIG_DEBUG_INFO_BTF enabled?"; \
		exit 1; \
	fi

# Clean build artifacts
clean:
	rm -f xid_tracer
	rm -f xid_bpfel.go xid_bpfel.o
	rm -f xid_bpfeb.go xid_bpfeb.o

# Install dependencies
deps:
	go mod download
	go install github.com/cilium/ebpf/cmd/bpf2go@latest

# Test (requires nvidia driver)
test:
	@if [ -d /sys/kernel/debug/tracing/events/nvidia ]; then \
		echo "NVIDIA tracepoint found: OK"; \
		ls -la /sys/kernel/debug/tracing/events/nvidia/; \
	else \
		echo "WARNING: NVIDIA tracepoint not found"; \
		echo "Is the nvidia driver loaded? Check: lsmod | grep nvidia"; \
	fi
