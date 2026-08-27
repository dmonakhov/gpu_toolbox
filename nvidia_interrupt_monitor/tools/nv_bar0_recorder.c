/*
 * nv_bar0_recorder: per-GPU high-rate (10-100 kHz) sampler of the
 * NVIDIA GPU interrupt aggregator (INTR_LEAF[0..15] pending) and
 * MSI-X PBA via BAR0 mmap. Strictly read-only by default.
 *
 * Use it to characterize a GPU's interrupt-delivery behavior under a
 * chosen workload: which leaf bits fire, at what rate, how long they
 * stay pending, and whether any bit is latched (stuck pending because
 * the MSI-X message was lost - see the project README).
 *
 * Everything below is computed offline from this recorder's output by
 * nv_bar0_analyzer.py:
 *   - engine/bit mapping: rising-edge counts under single-engine
 *     workloads identify which LEAF[i].bit belongs to which engine
 *   - per-bit occupancy (duty cycle of pending) and edge rates
 *   - latched-bit detection: occupancy ~100% with zero edges
 *   - interrupt latency studies: rising-edge timestamps can be joined
 *     against kernel-side ISR probe timestamps (e.g. bpftrace on the
 *     driver ISR) since both use CLOCK_MONOTONIC
 *
 * Threading model:
 *   - One pthread per GPU. Each pinned to its own CPU
 *     (--cpu-base + gpu_idx) and optionally SCHED_FIFO. This is
 *     how 100 kHz on 8 GPUs is reached: serial sampling can't
 *     keep up because each MMIO read takes ~1 us.
 *   - Main thread enumerates GPUs, snapshots LEAF_EN_SET / MSI-X
 *     table once at start (written to manifest.json), spawns
 *     samplers, waits for SIGINT or --duration, then joins.
 *
 * Modes:
 *   --mode raw       (default): stream every sample to a per-GPU
 *                    binary file. ~8 MB/s per GPU at 100 kHz x 80 B
 *                    sample. Bulky but lossless.
 *   --mode aggregate (long-running deployment): keep counters in RAM,
 *                    flush a summary JSON line per window
 *                    (--agg-window-ms, default 100). ~1000x smaller
 *                    on disk; loses per-sample detail but keeps
 *                    occupancy + rising-edge counts.
 *
 * Source
 *  http://github.com/dmonakhov/gpu_toolbox/nvidia_interrupt_monitor
 * Build:
 *   gcc -O2 -Wall -pthread -o nv_bar0_recorder nv_bar0_recorder.c
 *
 * Source for register offsets: NVIDIA open-gpu-kernel-modules
 * dev_vm.h + intr_cpu_tu102.c. Layout exists on Turing and later;
 * validated on Hopper (H100/H200).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sched.h>
#include <signal.h>
#include <dirent.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ---- Register layout (Turing+; validated on Hopper) ---- */
#define VF_BASE            0x00b80000UL
#define VF_OFF_LEAF        0x00001000UL    /* LEAF[i].pending = +i*4 */
#define VF_OFF_LEAF_ENS    0x00001200UL
#define VF_OFF_LEAF_ENC    0x00001400UL    /* WO: write 1 to clear bit in EN mask */
#define VF_OFF_TOP         0x00001600UL
#define VF_OFF_MSIX_TBL    0x00010000UL    /* BAR0 + 0xb90000 */
#define VF_OFF_MSIX_PBA    0x00020000UL    /* BAR0 + 0xba0000 */

#define INTR_NUM_LEAVES    16
#define MSIX_NUM_VECTORS   9   /* vectors used on Hopper */

#define MAX_GPUS           16

/* PMC_BOOT_0 top byte = architecture id. The VF-block layout exists on
 * Turing and later; this tool is validated on Hopper. --force bypasses. */
static int arch_known(uint8_t id) {
    return id == 0x16 || id == 0x17 || id == 0x18 || id == 0x19;
}

/* ---- Sample record (raw mode), exactly 80 bytes packed ----
 *
 * pba_bits is vec 0..63 (one qword). Hopper uses 9 MSI-X vectors so
 * only bits 0..8 carry signal; bits above the table count read as
 * reserved-1s noise, so format version 2 keeps a single qword. */
struct sample_record {
    uint64_t ts_ns;                          /* 8  CLOCK_MONOTONIC */
    uint32_t intr_leaf[INTR_NUM_LEAVES];     /* 64 LEAF[0..15] pending */
    uint64_t pba_bits;                       /* 8  PBA bits 0..63 */
} __attribute__((packed));
typedef char _check_sample_size[
    (sizeof(struct sample_record) == 80) ? 1 : -1];
#define SAMPLE_FORMAT_VERSION 2

/* ---- Aggregate state (per GPU) ---- */
struct aggregate_state {
    uint64_t window_start_ns;
    uint64_t window_samples;
    /* Per-leaf: number of samples in window where LEAF[i] != 0. */
    uint64_t leaf_pending_count[INTR_NUM_LEAVES];
    /* Per-bit: number of samples in window where (LEAF[i] >> b) & 1. */
    uint64_t leaf_bit_pending_count[INTR_NUM_LEAVES][32];
    /* Per-bit: number of 0->1 transitions in window. */
    uint64_t leaf_bit_rising_count[INTR_NUM_LEAVES][32];
    /* Per-bit: number of 1->0 transitions in window. */
    uint64_t leaf_bit_falling_count[INTR_NUM_LEAVES][32];
    /* MSI-X PBA: per-vector samples where bit set. */
    uint64_t pba_bit_pending_count[MSIX_NUM_VECTORS];
    uint64_t pba_bit_rising_count[MSIX_NUM_VECTORS];
    /* Last sample for edge detection. */
    uint32_t last_leaf[INTR_NUM_LEAVES];
    uint64_t last_pba0;
    int      first_sample;
};

/* ---- Per-GPU context ---- */
struct gpu_ctx {
    char       bdf[16];
    int        gpu_idx;
    int        cpu_pin;             /* -1 = no pin */
    int        bar0_fd;
    /* Mapped regions: INTR aggregator and PBA. Each mmap is page-aligned. */
    void      *intr_map; size_t intr_map_len; char *intr_base;
    void      *pba_map;  size_t pba_map_len;  volatile uint64_t *pba_p;
    /* Initial snapshots (written to manifest at start). */
    uint32_t   pmc_boot_0;
    uint32_t   leaf_en_set_initial[INTR_NUM_LEAVES];
    uint32_t   msix_tbl_initial[MSIX_NUM_VECTORS * 4];
    /* Raw mode output. */
    int        raw_fd;
    char       raw_path[1024];
    void      *raw_buf;             /* fwrite buffer */
    FILE      *raw_fp;
    /* Aggregate mode output. */
    FILE      *agg_fp;
    char       agg_path[1024];
    struct aggregate_state agg;
    /* Diagnostics. Computed from actual sample timestamps (not from
     * loop-internal "we fell behind" branches, which over-count when
     * the target rate is unreachable). */
    uint64_t   samples_total;
    uint64_t   first_sample_ts_ns;       /* 0 until first sample taken */
    uint64_t   last_sample_ts_ns;
    uint64_t   max_gap_ns;               /* max inter-sample gap seen */
    uint64_t   gap_2x_period_count;      /* samples where gap > 2*period (jitter) */
    uint64_t   gap_10x_period_count;     /* samples where gap > 10*period (preempted) */
    pthread_t  thread;
};

static struct gpu_ctx g_gpus[MAX_GPUS];
static int g_n_gpus = 0;

/* ---- Knobs ---- */
static char  g_outdir[512] = "nv_bar0_recorder_out";
static char  g_manifest_path[640];
static int   g_rate_hz       = 10000;
static int   g_duration_sec  = 0;       /* 0 = run until SIGINT */
static int   g_cpu_base      = -1;      /* -1 = no pin; else cpu_base+gpu_idx */
static int   g_use_sched_fifo = 0;
static int   g_agg_window_ms = 100;
enum mode { MODE_RAW, MODE_AGG };
static enum mode g_mode      = MODE_RAW;
static int   g_raw_buf_bytes = 4 * 1024 * 1024;
static int   g_list_gpus     = 0;       /* --list-gpus: print BDFs and exit */
static int   g_force_arch    = 0;       /* --force: skip architecture check */
static int   g_enable_all_leaves = 0;   /* --enable-all-leaves: write LEAF_EN_SET[*]=0xffffffff */
static volatile sig_atomic_t g_should_stop = 0;

static void on_sigstop(int s) { (void)s; g_should_stop = 1; }

/* ---- Tiny utils ---- */
static void elog(const char *fmt, ...) {
    va_list ap;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    char tbuf[64];
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    fprintf(stderr, "[%s] ", tbuf);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static int mkdir_p(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t n = strlen(tmp);
    if (n > 0 && tmp[n-1] == '/') tmp[n-1] = 0;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) && errno != EEXIST) return -1;
    return 0;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *map_region_prot(int fd, unsigned long off, size_t len,
                              int prot,
                              unsigned long *within_out, size_t *map_len_out) {
    long page = sysconf(_SC_PAGESIZE);
    unsigned long aligned = off & ~(unsigned long)(page - 1);
    unsigned long within = off - aligned;
    size_t need = within + len;
    size_t map_len = ((need + page - 1) / page) * page;
    void *m = mmap(NULL, map_len, prot, MAP_SHARED, fd, aligned);
    if (m == MAP_FAILED) return NULL;
    *within_out = within;
    *map_len_out = map_len;
    return m;
}

static void *map_region(int fd, unsigned long off, size_t len,
                        unsigned long *within_out, size_t *map_len_out) {
    return map_region_prot(fd, off, len, PROT_READ, within_out, map_len_out);
}

/* ---- GPU setup ---- */
static int snapshot_initial_state(struct gpu_ctx *g) {
    /* PMC_BOOT_0 */
    unsigned long within; size_t maplen;
    void *m = map_region(g->bar0_fd, 0x0, 4, &within, &maplen);
    if (!m) return -1;
    g->pmc_boot_0 = *(volatile uint32_t *)((char *)m + within);
    munmap(m, maplen);
    /* LEAF_EN_SET[0..15]: from already-mapped intr_base */
    volatile uint32_t *leaf_ens =
        (volatile uint32_t *)(g->intr_base + (VF_OFF_LEAF_ENS - VF_OFF_LEAF));
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        g->leaf_en_set_initial[i] = leaf_ens[i];
    }
    /* MSI-X table (9 vectors x 16 bytes). One separate mmap. */
    unsigned long tbl_off = VF_BASE + VF_OFF_MSIX_TBL;
    void *tbl = map_region(g->bar0_fd, tbl_off,
                           MSIX_NUM_VECTORS * 16, &within, &maplen);
    if (tbl) {
        volatile uint32_t *t = (volatile uint32_t *)((char *)tbl + within);
        for (int i = 0; i < MSIX_NUM_VECTORS * 4; i++) {
            g->msix_tbl_initial[i] = t[i];
        }
        munmap(tbl, maplen);
    }
    return 0;
}

static int gpu_open(struct gpu_ctx *g, const char *bdf, int gpu_idx) {
    memset(g, 0, sizeof(*g));
    snprintf(g->bdf, sizeof(g->bdf), "%s", bdf);
    g->gpu_idx = gpu_idx;
    g->bar0_fd = -1;
    g->raw_fd = -1;
    g->cpu_pin = (g_cpu_base >= 0) ? (g_cpu_base + gpu_idx) : -1;

    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int open_flags = (g_enable_all_leaves ? O_RDWR : O_RDONLY) | O_SYNC;
    int fd = open(path, open_flags);
    if (fd < 0) { elog("ERROR open(%s): %s", path, strerror(errno)); return -1; }

    /* Verify architecture via PMC_BOOT_0 BEFORE we map the aggregator. */
    unsigned long within; size_t maplen;
    void *m = map_region(fd, 0x0, 4, &within, &maplen);
    if (!m) { close(fd); return -1; }
    uint32_t pmc = *(volatile uint32_t *)((char *)m + within);
    munmap(m, maplen);
    if (pmc == 0 || pmc == 0xFFFFFFFF) {
        elog("SKIP %s: PMC_BOOT_0=0x%08x - BAR0 MMIO read broken", bdf, pmc);
        close(fd); return -1;
    }
    if (!arch_known((pmc >> 24) & 0xFF) && !g_force_arch) {
        elog("SKIP %s: PMC_BOOT_0=0x%08x unknown arch (use --force)", bdf, pmc);
        close(fd); return -1;
    }

    /* INTR aggregator: 0xb81000..0xb81614 (LEAF, LEAF_EN_SET, LEAF_EN_CLEAR, TOP).
     * When --enable-all-leaves is set we map RW so we can write
     * LEAF_EN_SET / LEAF_EN_CLEAR; otherwise RO is sufficient. */
    unsigned long intr_off = VF_BASE + VF_OFF_LEAF;
    size_t intr_span = (VF_OFF_TOP + 16) - VF_OFF_LEAF;
    int intr_prot = PROT_READ | (g_enable_all_leaves ? PROT_WRITE : 0);
    void *im = map_region_prot(fd, intr_off, intr_span, intr_prot,
                               &within, &maplen);
    if (!im) { elog("ERROR mmap INTR %s: %s", bdf, strerror(errno));
               close(fd); return -1; }
    g->intr_map = im; g->intr_map_len = maplen;
    g->intr_base = (char *)im + within;

    /* PBA: 24 bytes. */
    unsigned long pba_off = VF_BASE + VF_OFF_MSIX_PBA;
    void *pm = map_region(fd, pba_off, 24, &within, &maplen);
    if (pm) {
        g->pba_map = pm; g->pba_map_len = maplen;
        g->pba_p = (volatile uint64_t *)((char *)pm + within);
    } else {
        elog("WARN mmap PBA %s: %s (will read zeros)", bdf, strerror(errno));
        g->pba_p = NULL;
    }

    g->bar0_fd = fd;
    if (snapshot_initial_state(g) != 0) {
        elog("ERROR initial snapshot for %s", bdf);
    }

    /* Optionally enable all LEAF_EN_SET bits so engines that only
     * latch LEAF_PENDING when their EN bit is set become visible.
     * Original mask was snapshotted in leaf_en_set_initial; gpu_close
     * restores it. Sequence per leaf:
     *   1. write LEAF_EN_CLEAR[i] := 0xFFFFFFFF  (atomic disable-all)
     *   2. write LEAF_EN_SET[i]   := 0xFFFFFFFF  (atomic enable-all)
     * Non-destructive when paired with
     * the restore sequence in gpu_close. */
    if (g_enable_all_leaves) {
        volatile uint32_t *leaf_ens = (volatile uint32_t *)
            (g->intr_base + (VF_OFF_LEAF_ENS - VF_OFF_LEAF));
        volatile uint32_t *leaf_enc = (volatile uint32_t *)
            (g->intr_base + (VF_OFF_LEAF_ENC - VF_OFF_LEAF));
        for (int i = 0; i < INTR_NUM_LEAVES; i++) {
            leaf_enc[i] = 0xFFFFFFFFu;   /* disable all bits */
            leaf_ens[i] = 0xFFFFFFFFu;   /* enable all bits */
        }
        elog("OK enabled-all-leaves on %s (orig LEAF[0]=0x%08x restored on exit)",
             bdf, g->leaf_en_set_initial[0]);
    }

    elog("OK opened %s PMC_BOOT_0=0x%08x cpu_pin=%d",
         bdf, g->pmc_boot_0, g->cpu_pin);
    return 0;
}

static void gpu_close(struct gpu_ctx *g) {
    if (g->raw_fp) { fflush(g->raw_fp); fclose(g->raw_fp); g->raw_fp = NULL; }
    if (g->raw_buf) { free(g->raw_buf); g->raw_buf = NULL; }
    if (g->agg_fp) { fflush(g->agg_fp); fclose(g->agg_fp); g->agg_fp = NULL; }

    /* Restore LEAF_EN_SET to the original (snapshotted) mask if we
     * modified it. Sequence:
     *   1. LEAF_EN_CLEAR[i] := 0xFFFFFFFF (zero out)
     *   2. LEAF_EN_SET[i]   := original
     * Skipped if we never wrote it (g_enable_all_leaves not set) or
     * if the mapping is already gone. */
    if (g_enable_all_leaves && g->intr_base != NULL) {
        volatile uint32_t *leaf_ens = (volatile uint32_t *)
            (g->intr_base + (VF_OFF_LEAF_ENS - VF_OFF_LEAF));
        volatile uint32_t *leaf_enc = (volatile uint32_t *)
            (g->intr_base + (VF_OFF_LEAF_ENC - VF_OFF_LEAF));
        for (int i = 0; i < INTR_NUM_LEAVES; i++) {
            leaf_enc[i] = 0xFFFFFFFFu;
            leaf_ens[i] = g->leaf_en_set_initial[i];
        }
        elog("OK restored LEAF_EN_SET on %s", g->bdf);
    }

    if (g->intr_map) munmap(g->intr_map, g->intr_map_len);
    if (g->pba_map)  munmap(g->pba_map,  g->pba_map_len);
    if (g->bar0_fd >= 0) close(g->bar0_fd);
    memset(g, 0, sizeof(*g));
    g->bar0_fd = -1;
}

/* Walk /sys/bus/pci/devices, open each NVIDIA device, drop unsupported. */
static int discover_gpus(const char *only_bdf) {
    if (only_bdf && only_bdf[0]) {
        if (gpu_open(&g_gpus[g_n_gpus], only_bdf, g_n_gpus) == 0) g_n_gpus++;
        return g_n_gpus;
    }
    DIR *d = opendir("/sys/bus/pci/devices");
    if (!d) { elog("ERROR opendir: %s", strerror(errno)); return 0; }
    char (*entries)[16] = NULL;
    size_t ne = 0, cap = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        if (strlen(de->d_name) != 12) continue;
        char vp[256]; snprintf(vp, sizeof(vp),
                "/sys/bus/pci/devices/%s/vendor", de->d_name);
        FILE *fp = fopen(vp, "r");
        if (!fp) continue;
        char vbuf[16] = {0};
        if (!fgets(vbuf, sizeof(vbuf), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (strncasecmp(vbuf, "0x10de", 6) != 0) continue;
        if (ne == cap) { cap = cap ? cap*2 : 16;
                         entries = realloc(entries, cap*16); }
        snprintf(entries[ne++], 16, "%s", de->d_name);
    }
    closedir(d);
    for (size_t i = 0; i + 1 < ne; i++)
        for (size_t j = i + 1; j < ne; j++)
            if (strcmp(entries[i], entries[j]) > 0) {
                char tmp[16]; memcpy(tmp, entries[i], 16);
                memcpy(entries[i], entries[j], 16);
                memcpy(entries[j], tmp, 16);
            }
    for (size_t i = 0; i < ne && g_n_gpus < MAX_GPUS; i++) {
        if (gpu_open(&g_gpus[g_n_gpus], entries[i], g_n_gpus) == 0)
            g_n_gpus++;
    }
    free(entries);
    return g_n_gpus;
}

/* ---- Output setup ---- */
static int open_raw_output(struct gpu_ctx *g) {
    snprintf(g->raw_path, sizeof(g->raw_path),
             "%.500s/raw_trace/gpu_%.15s.bin", g_outdir, g->bdf);
    g->raw_fp = fopen(g->raw_path, "wb");
    if (!g->raw_fp) {
        elog("ERROR open raw %s: %s", g->raw_path, strerror(errno));
        return -1;
    }
    g->raw_buf = malloc(g_raw_buf_bytes);
    if (g->raw_buf) setvbuf(g->raw_fp, g->raw_buf, _IOFBF, g_raw_buf_bytes);
    return 0;
}

static int open_agg_output(struct gpu_ctx *g) {
    snprintf(g->agg_path, sizeof(g->agg_path),
             "%.500s/occupancy/gpu_%.15s.jsonl", g_outdir, g->bdf);
    g->agg_fp = fopen(g->agg_path, "w");
    if (!g->agg_fp) {
        elog("ERROR open agg %s: %s", g->agg_path, strerror(errno));
        return -1;
    }
    g->agg.first_sample = 1;
    return 0;
}

/* ---- Hot path: sample + dispatch ---- */
static inline void read_sample(struct gpu_ctx *g, uint64_t ts_ns,
                               struct sample_record *r) {
    r->ts_ns = ts_ns;
    volatile uint32_t *leaf =
        (volatile uint32_t *)(g->intr_base + 0x000);
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        r->intr_leaf[i] = leaf[i];
    }
    r->pba_bits = g->pba_p ? g->pba_p[0] : 0;
}

static void agg_flush_window(struct gpu_ctx *g, uint64_t ts_ns) {
    if (!g->agg_fp || g->agg.window_samples == 0) {
        g->agg.window_start_ns = ts_ns;
        return;
    }
    struct aggregate_state *a = &g->agg;
    fprintf(g->agg_fp, "{\"ts_start_ns\":%" PRIu64
                       ",\"ts_end_ns\":%" PRIu64
                       ",\"samples\":%" PRIu64 ",\"leaf_occ\":[",
            a->window_start_ns, ts_ns, a->window_samples);
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        fprintf(g->agg_fp, "%s%" PRIu64,
                i ? "," : "", a->leaf_pending_count[i]);
    }
    fprintf(g->agg_fp, "],\"pba_occ\":[");
    for (int v = 0; v < MSIX_NUM_VECTORS; v++) {
        fprintf(g->agg_fp, "%s%" PRIu64,
                v ? "," : "", a->pba_bit_pending_count[v]);
    }
    fprintf(g->agg_fp, "],\"pba_rising\":[");
    for (int v = 0; v < MSIX_NUM_VECTORS; v++) {
        fprintf(g->agg_fp, "%s%" PRIu64,
                v ? "," : "", a->pba_bit_rising_count[v]);
    }
    fprintf(g->agg_fp, "],\"leaf_bit_rising\":{");
    int first = 1;
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        for (int b = 0; b < 32; b++) {
            if (a->leaf_bit_rising_count[i][b] == 0) continue;
            fprintf(g->agg_fp, "%s\"%d.%d\":%" PRIu64,
                    first ? "" : ",", i, b,
                    a->leaf_bit_rising_count[i][b]);
            first = 0;
        }
    }
    fprintf(g->agg_fp, "},\"leaf_bit_occ\":{");
    first = 1;
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        for (int b = 0; b < 32; b++) {
            if (a->leaf_bit_pending_count[i][b] == 0) continue;
            fprintf(g->agg_fp, "%s\"%d.%d\":%" PRIu64,
                    first ? "" : ",", i, b,
                    a->leaf_bit_pending_count[i][b]);
            first = 0;
        }
    }
    fprintf(g->agg_fp, "}}\n");
    /* Reset counters for the next window. */
    a->window_start_ns = ts_ns;
    a->window_samples = 0;
    memset(a->leaf_pending_count, 0, sizeof(a->leaf_pending_count));
    memset(a->leaf_bit_pending_count, 0, sizeof(a->leaf_bit_pending_count));
    memset(a->leaf_bit_rising_count, 0, sizeof(a->leaf_bit_rising_count));
    memset(a->leaf_bit_falling_count, 0, sizeof(a->leaf_bit_falling_count));
    memset(a->pba_bit_pending_count, 0, sizeof(a->pba_bit_pending_count));
    memset(a->pba_bit_rising_count, 0, sizeof(a->pba_bit_rising_count));
}

static inline void agg_accumulate(struct gpu_ctx *g,
                                  const struct sample_record *r) {
    struct aggregate_state *a = &g->agg;
    if (a->first_sample) {
        a->window_start_ns = r->ts_ns;
        a->first_sample = 0;
        for (int i = 0; i < INTR_NUM_LEAVES; i++) {
            a->last_leaf[i] = r->intr_leaf[i];
        }
        a->last_pba0 = r->pba_bits;
        return; /* don't count first sample (no prev for edges) */
    }
    a->window_samples++;
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        uint32_t v = r->intr_leaf[i];
        if (v) a->leaf_pending_count[i]++;
        uint32_t rising  = v & ~a->last_leaf[i];
        uint32_t falling = ~v & a->last_leaf[i];
        for (int b = 0; b < 32; b++) {
            if ((v >> b) & 1) a->leaf_bit_pending_count[i][b]++;
            if ((rising  >> b) & 1) a->leaf_bit_rising_count[i][b]++;
            if ((falling >> b) & 1) a->leaf_bit_falling_count[i][b]++;
        }
        a->last_leaf[i] = v;
    }
    uint64_t pba0 = r->pba_bits;
    uint64_t pba_rising = pba0 & ~a->last_pba0;
    for (int v = 0; v < MSIX_NUM_VECTORS; v++) {
        if ((pba0 >> v) & 1ULL) a->pba_bit_pending_count[v]++;
        if ((pba_rising >> v) & 1ULL) a->pba_bit_rising_count[v]++;
    }
    a->last_pba0 = pba0;
}

/* ---- Per-GPU sampling thread ---- */
static void *sampler_thread(void *arg) {
    struct gpu_ctx *g = (struct gpu_ctx *)arg;
    if (g->cpu_pin >= 0) {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(g->cpu_pin, &s);
        if (sched_setaffinity(0, sizeof(s), &s) != 0) {
            elog("WARN setaffinity gpu_idx=%d cpu=%d: %s",
                 g->gpu_idx, g->cpu_pin, strerror(errno));
        }
    }
    if (g_use_sched_fifo) {
        struct sched_param sp = { .sched_priority = 50 };
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
            elog("WARN SCHED_FIFO gpu_idx=%d: %s",
                 g->gpu_idx, strerror(errno));
        }
    }

    uint64_t period_ns = 1000000000ULL / (uint64_t)g_rate_hz;
    uint64_t agg_window_ns = (uint64_t)g_agg_window_ms * 1000000ULL;
    uint64_t last_agg_flush = now_ns();

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    struct sample_record rec;

    while (!g_should_stop) {
        if (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL) != 0) {
            if (errno == EINTR) continue;
        }
        uint64_t ts = now_ns();
        read_sample(g, ts, &rec);

        if (g_mode == MODE_RAW) {
            if (fwrite(&rec, sizeof(rec), 1, g->raw_fp) != 1) {
                elog("WARN raw fwrite gpu_idx=%d: %s",
                     g->gpu_idx, strerror(errno));
            }
        } else {
            agg_accumulate(g, &rec);
            if (ts - last_agg_flush >= agg_window_ns) {
                agg_flush_window(g, ts);
                last_agg_flush = ts;
            }
        }
        /* Track real inter-sample timing from the actual ts stream.
         * This is the ground truth -- the loop's "we fell behind"
         * heuristic over-counts when the target rate is unreachable. */
        if (g->first_sample_ts_ns == 0) {
            g->first_sample_ts_ns = ts;
        } else {
            uint64_t gap = ts - g->last_sample_ts_ns;
            if (gap > g->max_gap_ns) g->max_gap_ns = gap;
            if (gap > 2  * period_ns) g->gap_2x_period_count++;
            if (gap > 10 * period_ns) g->gap_10x_period_count++;
        }
        g->last_sample_ts_ns = ts;
        g->samples_total++;

        /* Advance next tick. If we fell behind by >1 period, skip
         * ahead so the loop doesn't try to catch up infinitely. */
        uint64_t next_ns = (uint64_t)next.tv_sec * 1000000000ULL
                         + (uint64_t)next.tv_nsec + period_ns;
        if (ts > next_ns + period_ns) {
            next_ns = ts + period_ns;
        }
        next.tv_sec  = next_ns / 1000000000ULL;
        next.tv_nsec = next_ns % 1000000000ULL;
    }

    /* Final flush. */
    if (g_mode == MODE_RAW) {
        fflush(g->raw_fp);
    } else {
        agg_flush_window(g, now_ns());
        fflush(g->agg_fp);
    }
    return NULL;
}

/* ---- Manifest ---- */
static void emit_hex_array(FILE *fp, const uint32_t *v, int n) {
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%s\"0x%08x\"", i ? "," : "", v[i]);
    }
}

static void write_manifest(void) {
    FILE *fp = fopen(g_manifest_path, "w");
    if (!fp) { elog("WARN manifest open: %s", strerror(errno)); return; }
    char host[256] = "unknown";
    gethostname(host, sizeof(host));
    fprintf(fp,
        "{\n"
        "  \"tool\": \"nv_bar0_recorder\",\n"
        "  \"version\": 1,\n"
        "  \"host\": \"%s\",\n"
        "  \"start_ts_unix\": %ld,\n"
        "  \"rate_hz\": %d,\n"
        "  \"mode\": \"%s\",\n"
        "  \"duration_sec\": %d,\n"
        "  \"agg_window_ms\": %d,\n"
        "  \"sched_fifo\": %s,\n"
        "  \"cpu_base\": %d,\n"
        "  \"sample_record_bytes\": %zu,\n"
        "  \"sample_format_version\": %d,\n"
        "  \"gpus\": [\n",
        host, (long)time(NULL), g_rate_hz,
        g_mode == MODE_RAW ? "raw" : "aggregate",
        g_duration_sec, g_agg_window_ms,
        g_use_sched_fifo ? "true" : "false",
        g_cpu_base, sizeof(struct sample_record),
        SAMPLE_FORMAT_VERSION);
    for (int i = 0; i < g_n_gpus; i++) {
        struct gpu_ctx *g = &g_gpus[i];
        fprintf(fp, "    {\"bdf\":\"%s\",\"gpu_idx\":%d,\"cpu_pin\":%d,"
                    "\"pmc_boot_0\":\"0x%08x\","
                    "\"leaf_en_set_initial\":[",
                g->bdf, g->gpu_idx, g->cpu_pin, g->pmc_boot_0);
        emit_hex_array(fp, g->leaf_en_set_initial, INTR_NUM_LEAVES);
        fprintf(fp, "],\"msix_tbl_initial\":[");
        emit_hex_array(fp, g->msix_tbl_initial, MSIX_NUM_VECTORS * 4);
        fprintf(fp, "]}%s\n", (i + 1 < g_n_gpus) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

static void write_run_report(void) {
    char path[640];
    snprintf(path, sizeof(path), "%s/run_report.json", g_outdir);
    FILE *fp = fopen(path, "w");
    if (!fp) { elog("WARN report open: %s", strerror(errno)); return; }
    fprintf(fp, "{\n  \"gpus\": [\n");
    for (int i = 0; i < g_n_gpus; i++) {
        struct gpu_ctx *g = &g_gpus[i];
        /* Read final LEAF_EN_SET to verify it didn't change. */
        uint32_t leaf_en_final[INTR_NUM_LEAVES];
        volatile uint32_t *leaf_ens =
            (volatile uint32_t *)(g->intr_base + (VF_OFF_LEAF_ENS - VF_OFF_LEAF));
        for (int k = 0; k < INTR_NUM_LEAVES; k++)
            leaf_en_final[k] = leaf_ens[k];
        int changed = 0;
        for (int k = 0; k < INTR_NUM_LEAVES; k++)
            if (leaf_en_final[k] != g->leaf_en_set_initial[k]) changed = 1;
        double elapsed_s = 0.0, observed_hz = 0.0;
        if (g->first_sample_ts_ns && g->last_sample_ts_ns > g->first_sample_ts_ns) {
            elapsed_s = (double)(g->last_sample_ts_ns - g->first_sample_ts_ns) / 1e9;
            if (elapsed_s > 0.0)
                observed_hz = (double)(g->samples_total - 1) / elapsed_s;
        }
        double max_gap_us = (double)g->max_gap_ns / 1000.0;
        double p_jitter_2x  = g->samples_total
            ? (double)g->gap_2x_period_count / (double)g->samples_total : 0.0;
        double p_jitter_10x = g->samples_total
            ? (double)g->gap_10x_period_count / (double)g->samples_total : 0.0;
        fprintf(fp, "    {\"bdf\":\"%s\",\"samples_total\":%" PRIu64
                    ",\"elapsed_s\":%.3f,\"observed_hz\":%.1f"
                    ",\"max_gap_us\":%.2f"
                    ",\"frac_gap_above_2x_period\":%.5f"
                    ",\"frac_gap_above_10x_period\":%.5f"
                    ",\"leaf_en_set_changed\":%s,\"leaf_en_set_final\":[",
                g->bdf, g->samples_total,
                elapsed_s, observed_hz, max_gap_us,
                p_jitter_2x, p_jitter_10x,
                changed ? "true" : "false");
        emit_hex_array(fp, leaf_en_final, INTR_NUM_LEAVES);
        fprintf(fp, "]}%s\n", (i + 1 < g_n_gpus) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --outdir <dir>        output directory (default ./nv_bar0_recorder_out)\n"
        "  --rate <hz>           sample rate per GPU (default 10000)\n"
        "  --duration <s>        run time, 0 = until SIGINT (default 0)\n"
        "  --mode raw|aggregate  output mode (default raw)\n"
        "  --agg-window-ms <ms>  aggregate flush interval (default 100)\n"
        "  --cpu-base <n>        pin gpu_idx=k to CPU n+k\n"
        "  --sched-fifo          SCHED_FIFO priority 50\n"
        "  --gpu <bdf>           restrict to one GPU (default: all NVIDIA GPUs)\n"
        "  --raw-buf-mb <n>      raw mode write buffer per GPU (default 4)\n"
        "  --list-gpus           enumerate GPU BDFs to stdout and exit\n"
        "  --force               scan unrecognized GPU architectures too\n"
        "  --enable-all-leaves   write LEAF_EN_SET[i]=0xffffffff for all i at\n"
        "                        startup; restore on exit. Use to probe engines\n"
        "                        whose LEAF_PENDING latch is gated on EN_SET.\n"
        "                        Requires write access to BAR0 (privileged).\n"
        "  --help                this help\n", prog);
}

int main(int argc, char **argv) {
    const char *only_bdf = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--outdir")  && i+1 < argc)
            snprintf(g_outdir, sizeof(g_outdir), "%s", argv[++i]);
        else if (!strcmp(a, "--rate")    && i+1 < argc) g_rate_hz = atoi(argv[++i]);
        else if (!strcmp(a, "--duration")&& i+1 < argc) g_duration_sec = atoi(argv[++i]);
        else if (!strcmp(a, "--mode")    && i+1 < argc) {
            const char *m = argv[++i];
            if      (!strcmp(m, "raw"))       g_mode = MODE_RAW;
            else if (!strcmp(m, "aggregate")) g_mode = MODE_AGG;
            else { fprintf(stderr, "bad --mode\n"); return 2; }
        }
        else if (!strcmp(a, "--agg-window-ms") && i+1 < argc)
            g_agg_window_ms = atoi(argv[++i]);
        else if (!strcmp(a, "--cpu-base") && i+1 < argc)
            g_cpu_base = atoi(argv[++i]);
        else if (!strcmp(a, "--sched-fifo")) g_use_sched_fifo = 1;
        else if (!strcmp(a, "--gpu") && i+1 < argc) only_bdf = argv[++i];
        else if (!strcmp(a, "--raw-buf-mb") && i+1 < argc)
            g_raw_buf_bytes = atoi(argv[++i]) * 1024 * 1024;
        else if (!strcmp(a, "--list-gpus")) g_list_gpus = 1;
        else if (!strcmp(a, "--force")) g_force_arch = 1;
        else if (!strcmp(a, "--enable-all-leaves")) g_enable_all_leaves = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown: %s\n", a); usage(argv[0]); return 2; }
    }
    if (g_rate_hz < 1 || g_rate_hz > 2000000) {
        fprintf(stderr, "ERROR --rate out of range\n"); return 2;
    }

    /* --list-gpus: enumerate supported BDFs to stdout, exit. No outdir,
     * no manifest, no threads. Handy for wrapper scripts: read() on
     * resource0 is broken on some kernels, mmap-based detection is not. */
    if (g_list_gpus) {
        if (discover_gpus(only_bdf) == 0) {
            fprintf(stderr, "no supported NVIDIA GPUs\n");
            return 4;
        }
        for (int i = 0; i < g_n_gpus; i++) {
            printf("%s\n", g_gpus[i].bdf);
        }
        fflush(stdout);
        for (int i = 0; i < g_n_gpus; i++) gpu_close(&g_gpus[i]);
        return 0;
    }

    if (mkdir_p(g_outdir) != 0) { elog("ERROR mkdir %s", g_outdir); return 3; }
    char sub[640];
    snprintf(sub, sizeof(sub), "%s/raw_trace", g_outdir);
    if (g_mode == MODE_RAW)  mkdir_p(sub);
    snprintf(sub, sizeof(sub), "%s/occupancy", g_outdir);
    if (g_mode == MODE_AGG)  mkdir_p(sub);
    snprintf(g_manifest_path, sizeof(g_manifest_path),
             "%s/manifest.json", g_outdir);

    signal(SIGINT,  on_sigstop);
    signal(SIGTERM, on_sigstop);

    if (discover_gpus(only_bdf) == 0) {
        elog("ERROR no supported NVIDIA GPUs"); return 4;
    }
    elog("discovered %d GPU(s), mode=%s rate=%d Hz",
         g_n_gpus, g_mode == MODE_RAW ? "raw" : "aggregate", g_rate_hz);

    for (int i = 0; i < g_n_gpus; i++) {
        if (g_mode == MODE_RAW) {
            if (open_raw_output(&g_gpus[i]) != 0) {
                for (int j = 0; j <= i; j++) gpu_close(&g_gpus[j]);
                return 5;
            }
        } else {
            if (open_agg_output(&g_gpus[i]) != 0) {
                for (int j = 0; j <= i; j++) gpu_close(&g_gpus[j]);
                return 5;
            }
        }
    }

    write_manifest();

    uint64_t start_ns = now_ns();
    for (int i = 0; i < g_n_gpus; i++) {
        if (pthread_create(&g_gpus[i].thread, NULL,
                           sampler_thread, &g_gpus[i]) != 0) {
            elog("ERROR pthread_create gpu_idx=%d: %s", i, strerror(errno));
            g_should_stop = 1;
        }
    }

    /* Main thread: wait for duration or SIGINT, then signal stop. */
    while (!g_should_stop) {
        if (g_duration_sec > 0 &&
            (now_ns() - start_ns) >= (uint64_t)g_duration_sec * 1000000000ULL)
            break;
        /* Heartbeat every 30 s. Report observed per-GPU effective Hz
         * (ground truth, derived from sample timestamps -- not from
         * the loop's miss heuristic). */
        double elapsed = (double)(now_ns() - start_ns) / 1e9;
        uint64_t total = 0;
        double min_hz = 0, max_hz = 0;
        int first = 1;
        for (int i = 0; i < g_n_gpus; i++) {
            total += g_gpus[i].samples_total;
            double per = g_gpus[i].samples_total / (elapsed > 0 ? elapsed : 1);
            if (first) { min_hz = max_hz = per; first = 0; }
            if (per < min_hz) min_hz = per;
            if (per > max_hz) max_hz = per;
        }
        elog("heartbeat: elapsed=%.1f s samples=%" PRIu64
             " per_gpu_hz=[%.0f..%.0f]",
             elapsed, total, min_hz, max_hz);
        sleep(30);
    }
    g_should_stop = 1;

    for (int i = 0; i < g_n_gpus; i++) {
        pthread_join(g_gpus[i].thread, NULL);
    }

    write_run_report();
    elog("stopping: total_samples=%" PRIu64,
         ({ uint64_t t = 0; for (int i = 0; i < g_n_gpus; i++) t += g_gpus[i].samples_total; t; }));

    for (int i = 0; i < g_n_gpus; i++) gpu_close(&g_gpus[i]);
    return 0;
}
