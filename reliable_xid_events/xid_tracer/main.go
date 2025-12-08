//go:build linux

// xid_tracer traces NVIDIA GPU XID errors using the nvidia:nvidia_dev_xid kernel tracepoint.
// Requires: Linux kernel with nvidia driver loaded, CAP_BPF or root privileges.
//
// Usage:
//   go generate ./...
//   go build -o xid_tracer .
//   sudo ./xid_tracer

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"github.com/cilium/ebpf/rlimit"
)

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target amd64 xid ./xid.bpf.c

// XidEvent mirrors the xid_event struct from the eBPF program
type XidEvent struct {
	ErrorCode uint32
	PciDev    [16]byte
	Msg       [256]byte
}

func main() {
	// Allow the current process to lock memory for eBPF resources
	if err := rlimit.RemoveMemlock(); err != nil {
		log.Fatalf("Failed to remove memlock limit: %v", err)
	}

	// Check if nvidia tracepoint exists
	tracepointPath := "/sys/kernel/debug/tracing/events/nvidia/nvidia_dev_xid"
	if _, err := os.Stat(tracepointPath); os.IsNotExist(err) {
		log.Fatalf("NVIDIA tracepoint not found at %s. Is the nvidia driver loaded?", tracepointPath)
	}

	// Load pre-compiled eBPF programs and maps
	objs := xidObjects{}
	if err := loadXidObjects(&objs, nil); err != nil {
		log.Fatalf("Loading eBPF objects: %v", err)
	}
	defer objs.Close()

	// Attach eBPF program to the nvidia:nvidia_dev_xid tracepoint
	tp, err := link.Tracepoint("nvidia", "nvidia_dev_xid", objs.HandleNvidiaXid, nil)
	if err != nil {
		log.Fatalf("Attaching tracepoint: %v", err)
	}
	defer tp.Close()

	// Open perf event reader to receive events from kernel
	rd, err := perf.NewReader(objs.Events, os.Getpagesize()*8)
	if err != nil {
		log.Fatalf("Creating perf event reader: %v", err)
	}
	defer rd.Close()

	// Handle signals for graceful shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	log.Println("Waiting for NVIDIA XID events... (press Ctrl+C to exit)")
	log.Println("Tracepoint: nvidia:nvidia_dev_xid")

	go func() {
		<-sig
		log.Println("Received signal, exiting...")
		rd.Close()
	}()

	// Read events loop
	for {
		record, err := rd.Read()
		if err != nil {
			if errors.Is(err, perf.ErrClosed) {
				return
			}
			log.Printf("Reading perf event: %v", err)
			continue
		}

		if record.LostSamples > 0 {
			log.Printf("WARNING: Lost %d samples due to perf buffer overflow", record.LostSamples)
			continue
		}

		// Parse the event
		var event XidEvent
		if err := binary.Read(bytes.NewReader(record.RawSample), binary.LittleEndian, &event); err != nil {
			log.Printf("Parsing event: %v", err)
			continue
		}

		// Extract null-terminated strings
		pciDev := nullTerminatedString(event.PciDev[:])
		msg := nullTerminatedString(event.Msg[:])

		// Print the XID event
		fmt.Printf("XID event received: code=%d pci=%s msg=%s\n", event.ErrorCode, pciDev, msg)
	}
}

// nullTerminatedString extracts a string from a null-terminated byte slice
func nullTerminatedString(b []byte) string {
	if idx := bytes.IndexByte(b, 0); idx != -1 {
		return string(b[:idx])
	}
	return string(b)
}
