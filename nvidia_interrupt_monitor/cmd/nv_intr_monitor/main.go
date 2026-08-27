// nv_intr_monitor: per-node daemon that watches NVIDIA GPU interrupt
// aggregator (INTR_LEAF) registers over BAR0 and detects latched
// (stuck-pending) interrupt vectors -- the fingerprint of an MSI-X
// message lost in transit (hypervisor live update / live migration
// pause, host kexec, AER recovery, ...).
//
// On a healthy GPU a pending bit is drained by the driver ISR within
// microseconds. A bit that stays pending across -confirm consecutive
// scans (default 3 scans x 30s apart) while enabled in LEAF_EN_SET is a
// lost interrupt: the ISR will never run for it and completions on that
// vector are silently gone. The workload symptom is long kernel-dispatch
// stalls with zero errors anywhere (no Xid, clean dmesg, clean ECC).
//
// Designed to run as a privileged Kubernetes DaemonSet (see
// deploy/daemonset.yaml) or directly as a systemd service. Strictly
// read-only: a handful of 32-bit MMIO reads per GPU per scan.
//
// Endpoints (default :9834):
//
//	/metrics  Prometheus text exposition
//	/healthz  200 "ok" when no latched bits, 503 "stuck:N" otherwise
//	/status   JSON snapshot of the current state
//
// Register layout: Turing and later, validated on Hopper (H100/H200).
// Source: NVIDIA open-gpu-kernel-modules dev_vm.h + intr_cpu_tu102.c.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

const (
	vfBase       = 0x00b80000
	vfOffLeaf    = 0x00001000 // LEAF[i].pending = +i*4
	vfOffLeafEns = 0x00001200 // LEAF_EN_SET[i]  = +i*4
	vfOffTop     = 0x00001600
	vfOffMsixPba = 0x00020000
	numLeaves    = 16
	pageSize     = 4096
)

// PMC_BOOT_0 top byte = architecture id. The VF-block layout exists on
// Turing and later; validated on Hopper. -force bypasses the check.
var knownArchs = map[byte]string{
	0x16: "Turing",
	0x17: "Ampere",
	0x18: "Hopper",
	0x19: "Ada",
}

type gpuScan struct {
	BDF     string            `json:"bdf"`
	PMC     uint32            `json:"pmc_boot_0"`
	Arch    string            `json:"arch"`
	Top     uint32            `json:"top"`
	Leaf    [numLeaves]uint32 `json:"leaf_pending"`
	LeafEns [numLeaves]uint32 `json:"leaf_en_set"`
	PBA     uint64            `json:"msix_pba"`
	Err     string            `json:"error,omitempty"`
}

type stuckBit struct {
	BDF       string    `json:"bdf"`
	Leaf      int       `json:"leaf"`
	Bit       int       `json:"bit"`
	Enabled   bool      `json:"enabled"`
	Consec    int       `json:"consecutive_scans"`
	FirstSeen time.Time `json:"first_seen"`
	Latched   bool      `json:"latched"` // Consec >= confirm threshold
}

type monitor struct {
	mu          sync.Mutex
	confirm     int
	force       bool
	scansTotal  uint64
	scanErrors  uint64
	lastScan    time.Time
	gpus        []gpuScan
	pending     map[string]*stuckBit // key: bdf/leaf/bit
	everLatched uint64
}

func key(bdf string, leaf, bit int) string {
	return fmt.Sprintf("%s/%d/%d", bdf, leaf, bit)
}

func leafRole(i int) string {
	switch {
	case i <= 1:
		return "ENGINE_NOTIFICATION_NONSTALL"
	case i <= 3:
		return "UVM_OWNED"
	case i <= 5:
		return "UVM_SHARED"
	case i <= 7:
		return "ENGINE_STALL"
	case i <= 9:
		return "RUNLIST"
	case i <= 11:
		return "RUNLIST_NOTIFICATION"
	default:
		return "unused"
	}
}

// readU32 reads a 32-bit MMIO register from a mapped region.
// atomic.LoadUint32 keeps the compiler from caching or tearing the read.
func readU32(m []byte, off int) uint32 {
	return atomic.LoadUint32((*uint32)(unsafe.Pointer(&m[off])))
}

func readU64(m []byte, off int) uint64 {
	return atomic.LoadUint64((*uint64)(unsafe.Pointer(&m[off])))
}

// mmapRegion maps len bytes at absolute BAR0 offset off (page-aligned
// internally). Returns the mapping and the offset of `off` within it.
func mmapRegion(fd int, off int64, length int) ([]byte, int, error) {
	aligned := off &^ (pageSize - 1)
	within := int(off - aligned)
	mapLen := (within + length + pageSize - 1) &^ (pageSize - 1)
	m, err := syscall.Mmap(fd, aligned, mapLen,
		syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, 0, err
	}
	return m, within, nil
}

// discoverGPUs returns BDFs of NVIDIA PCI devices, sorted.
func discoverGPUs() ([]string, error) {
	ents, err := os.ReadDir("/sys/bus/pci/devices")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range ents {
		v, err := os.ReadFile(filepath.Join("/sys/bus/pci/devices",
			e.Name(), "vendor"))
		if err != nil {
			continue
		}
		if strings.TrimSpace(string(v)) == "0x10de" {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

// scanGPU reads the interrupt state of one GPU. Read-only.
func scanGPU(bdf string, force bool) gpuScan {
	s := gpuScan{BDF: bdf}
	path := filepath.Join("/sys/bus/pci/devices", bdf, "resource0")
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_SYNC, 0)
	if err != nil {
		s.Err = fmt.Sprintf("open %s: %v", path, err)
		return s
	}
	defer syscall.Close(fd)

	// Sanity: PMC_BOOT_0 at BAR0+0 must be a real chip id.
	m0, w0, err := mmapRegion(fd, 0, 4)
	if err != nil {
		s.Err = fmt.Sprintf("mmap pmc: %v", err)
		return s
	}
	s.PMC = readU32(m0, w0)
	syscall.Munmap(m0)
	if s.PMC == 0 || s.PMC == 0xFFFFFFFF {
		s.Err = fmt.Sprintf("PMC_BOOT_0=0x%08x: BAR0 MMIO read broken", s.PMC)
		return s
	}
	arch := byte(s.PMC >> 24)
	name, ok := knownArchs[arch]
	if !ok {
		if !force {
			s.Err = fmt.Sprintf("PMC_BOOT_0=0x%08x: unknown arch 0x%02x (use -force)",
				s.PMC, arch)
			return s
		}
		name = fmt.Sprintf("unknown-0x%02x", arch)
	}
	s.Arch = name

	// Aggregator block: LEAF, LEAF_EN_SET, TOP live in one page.
	mi, wi, err := mmapRegion(fd, vfBase+vfOffLeaf, 0x700)
	if err != nil {
		s.Err = fmt.Sprintf("mmap aggregator: %v", err)
		return s
	}
	for i := 0; i < numLeaves; i++ {
		s.Leaf[i] = readU32(mi, wi+i*4)
		s.LeafEns[i] = readU32(mi, wi+(vfOffLeafEns-vfOffLeaf)+i*4)
	}
	s.Top = readU32(mi, wi+(vfOffTop-vfOffLeaf))
	syscall.Munmap(mi)

	// MSI-X PBA (informational; expected clean even when a leaf bit is
	// latched -- the loss is upstream of the PCIe MSI-X machinery).
	mp, wp, err := mmapRegion(fd, vfBase+vfOffMsixPba, 8)
	if err == nil {
		s.PBA = readU64(mp, wp)
		syscall.Munmap(mp)
	}
	return s
}

// update ingests one round of per-GPU scans and advances stuck tracking.
func (mon *monitor) update(scans []gpuScan) {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	mon.scansTotal++
	mon.lastScan = time.Now()
	mon.gpus = scans

	seen := map[string]bool{}
	for _, s := range scans {
		if s.Err != "" {
			mon.scanErrors++
			continue
		}
		for leaf := 0; leaf < numLeaves; leaf++ {
			v := s.Leaf[leaf]
			if v == 0 {
				continue
			}
			for bit := 0; bit < 32; bit++ {
				if v>>uint(bit)&1 == 0 {
					continue
				}
				k := key(s.BDF, leaf, bit)
				seen[k] = true
				enabled := s.LeafEns[leaf]>>uint(bit)&1 == 1
				sb, ok := mon.pending[k]
				if !ok {
					sb = &stuckBit{BDF: s.BDF, Leaf: leaf, Bit: bit,
						FirstSeen: time.Now()}
					mon.pending[k] = sb
				}
				sb.Consec++
				sb.Enabled = enabled
				if sb.Consec == mon.confirm && enabled {
					sb.Latched = true
					mon.everLatched++
					log.Printf("[WARN] latched interrupt: bdf=%s leaf=%d bit=%d role=%s pending_since=%s (%d consecutive scans)",
						s.BDF, leaf, bit, leafRole(leaf),
						sb.FirstSeen.Format(time.RFC3339), sb.Consec)
				}
			}
		}
	}
	// Bits that went away were drained by the ISR: normal operation.
	for k, sb := range mon.pending {
		if !seen[k] {
			if sb.Latched {
				log.Printf("[OK] latched bit cleared: bdf=%s leaf=%d bit=%d (was pending since %s)",
					sb.BDF, sb.Leaf, sb.Bit,
					sb.FirstSeen.Format(time.RFC3339))
			}
			delete(mon.pending, k)
		}
	}
}

func (mon *monitor) latchedBits() []*stuckBit {
	var out []*stuckBit
	for _, sb := range mon.pending {
		if sb.Latched {
			out = append(out, sb)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.BDF != b.BDF {
			return a.BDF < b.BDF
		}
		if a.Leaf != b.Leaf {
			return a.Leaf < b.Leaf
		}
		return a.Bit < b.Bit
	})
	return out
}

func popcount(v uint32) int {
	n := 0
	for ; v != 0; v &= v - 1 {
		n++
	}
	return n
}

func (mon *monitor) serveMetrics(w http.ResponseWriter, _ *http.Request) {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	var b strings.Builder
	fmt.Fprintf(&b, "# HELP nvidia_intr_monitor_scans_total Completed scan rounds.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_monitor_scans_total counter\n")
	fmt.Fprintf(&b, "nvidia_intr_monitor_scans_total %d\n", mon.scansTotal)
	fmt.Fprintf(&b, "# HELP nvidia_intr_monitor_scan_errors_total Per-GPU scan failures.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_monitor_scan_errors_total counter\n")
	fmt.Fprintf(&b, "nvidia_intr_monitor_scan_errors_total %d\n", mon.scanErrors)
	fmt.Fprintf(&b, "# HELP nvidia_intr_monitor_gpus GPUs scanned in the last round.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_monitor_gpus gauge\n")
	fmt.Fprintf(&b, "nvidia_intr_monitor_gpus %d\n", len(mon.gpus))

	fmt.Fprintf(&b, "# HELP nvidia_intr_leaf_pending_bits Popcount of INTR_LEAF pending bits (last scan).\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_leaf_pending_bits gauge\n")
	for _, s := range mon.gpus {
		if s.Err != "" {
			continue
		}
		for i := 0; i < numLeaves; i++ {
			if n := popcount(s.Leaf[i]); n > 0 {
				fmt.Fprintf(&b, "nvidia_intr_leaf_pending_bits{bdf=%q,leaf=\"%d\"} %d\n",
					s.BDF, i, n)
			}
		}
	}

	fmt.Fprintf(&b, "# HELP nvidia_intr_stuck_bit A latched (lost) interrupt vector. Value is 1.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_stuck_bit gauge\n")
	fmt.Fprintf(&b, "# HELP nvidia_intr_stuck_bit_age_seconds How long the bit has been pending.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_stuck_bit_age_seconds gauge\n")
	stuckPerGPU := map[string]int{}
	for _, sb := range mon.latchedBits() {
		stuckPerGPU[sb.BDF]++
		fmt.Fprintf(&b, "nvidia_intr_stuck_bit{bdf=%q,leaf=\"%d\",bit=\"%d\",role=%q} 1\n",
			sb.BDF, sb.Leaf, sb.Bit, leafRole(sb.Leaf))
		fmt.Fprintf(&b, "nvidia_intr_stuck_bit_age_seconds{bdf=%q,leaf=\"%d\",bit=\"%d\"} %.0f\n",
			sb.BDF, sb.Leaf, sb.Bit, time.Since(sb.FirstSeen).Seconds())
	}

	fmt.Fprintf(&b, "# HELP nvidia_intr_gpu_stuck 1 when the GPU has at least one latched interrupt vector.\n")
	fmt.Fprintf(&b, "# TYPE nvidia_intr_gpu_stuck gauge\n")
	for _, s := range mon.gpus {
		if s.Err != "" {
			continue
		}
		v := 0
		if stuckPerGPU[s.BDF] > 0 {
			v = 1
		}
		fmt.Fprintf(&b, "nvidia_intr_gpu_stuck{bdf=%q} %d\n", s.BDF, v)
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprint(w, b.String())
}

func (mon *monitor) serveHealthz(w http.ResponseWriter, _ *http.Request) {
	mon.mu.Lock()
	n := len(mon.latchedBits())
	mon.mu.Unlock()
	if n == 0 {
		fmt.Fprintln(w, "ok")
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	fmt.Fprintf(w, "stuck:%d\n", n)
}

func (mon *monitor) statusJSON() ([]byte, error) {
	mon.mu.Lock()
	defer mon.mu.Unlock()
	host, _ := os.Hostname()
	return json.MarshalIndent(map[string]interface{}{
		"host":        host,
		"last_scan":   mon.lastScan,
		"scans_total": mon.scansTotal,
		"scan_errors": mon.scanErrors,
		"gpus":        mon.gpus,
		"latched":     mon.latchedBits(),
	}, "", "  ")
}

func (mon *monitor) serveStatus(w http.ResponseWriter, _ *http.Request) {
	j, err := mon.statusJSON()
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(j)
}

func (mon *monitor) scanOnce() {
	bdfs, err := discoverGPUs()
	if err != nil {
		log.Printf("[ERROR] discover GPUs: %v", err)
		mon.mu.Lock()
		mon.scanErrors++
		mon.mu.Unlock()
		return
	}
	if len(bdfs) == 0 {
		log.Printf("[WARN] no NVIDIA GPUs found in /sys/bus/pci/devices")
		mon.mu.Lock()
		mon.scanErrors++
		mon.mu.Unlock()
	}
	scans := make([]gpuScan, 0, len(bdfs))
	for _, bdf := range bdfs {
		s := scanGPU(bdf, mon.force)
		if s.Err != "" {
			log.Printf("[ERROR] scan %s: %s", bdf, s.Err)
		}
		scans = append(scans, s)
	}
	mon.update(scans)
}

func main() {
	interval := flag.Duration("interval", 30*time.Second,
		"time between scans")
	confirm := flag.Int("confirm", 3,
		"consecutive scans a bit must stay pending to count as latched")
	listen := flag.String("listen", ":9834",
		"metrics/health listen address")
	force := flag.Bool("force", false,
		"scan unrecognized GPU architectures too")
	oneshot := flag.Bool("oneshot", false,
		"run -confirm scans -oneshot-interval apart, print JSON status, "+
			"exit 0 (healthy), 1 (latched bits), 2 (scan errors)")
	oneshotInterval := flag.Duration("oneshot-interval", 1*time.Second,
		"interval between scans in -oneshot mode")
	stateFile := flag.String("state", "",
		"optional path to write JSON status after every scan")
	flag.Parse()

	if *confirm < 1 {
		*confirm = 1
	}
	mon := &monitor{
		confirm: *confirm,
		force:   *force,
		pending: map[string]*stuckBit{},
	}

	if *oneshot {
		for i := 0; i < *confirm; i++ {
			if i > 0 {
				time.Sleep(*oneshotInterval)
			}
			mon.scanOnce()
		}
		j, _ := mon.statusJSON()
		fmt.Println(string(j))
		if len(mon.latchedBits()) > 0 {
			os.Exit(1)
		}
		if mon.scanErrors > 0 {
			os.Exit(2)
		}
		return
	}

	http.HandleFunc("/metrics", mon.serveMetrics)
	http.HandleFunc("/healthz", mon.serveHealthz)
	http.HandleFunc("/status", mon.serveStatus)
	go func() {
		log.Printf("listening on %s (/metrics /healthz /status)", *listen)
		if err := http.ListenAndServe(*listen, nil); err != nil {
			log.Fatalf("http: %v", err)
		}
	}()

	log.Printf("nv_intr_monitor starting: interval=%s confirm=%d", *interval, *confirm)
	for {
		mon.scanOnce()
		if *stateFile != "" {
			if j, err := mon.statusJSON(); err == nil {
				tmp := *stateFile + ".tmp"
				if os.WriteFile(tmp, j, 0644) == nil {
					os.Rename(tmp, *stateFile)
				}
			}
		}
		time.Sleep(*interval)
	}
}
