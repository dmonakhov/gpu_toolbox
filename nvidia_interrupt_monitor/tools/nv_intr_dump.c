/*
 * nv_intr_dump: dump NVIDIA GPU interrupt-delivery state via BAR0 mmap
 * and detect latched (stuck-pending) interrupt vectors.
 * Source: http://github.com/dmonakhov/gpu_toolbox/nvidia_interrupt_monitor
 *
 * Why
 * ----------
 * MSI-X interrupts are posted memory writes, not wires. Any event that
 * pauses or replays PCIe delivery for a moment (hypervisor live update /
 * live migration, host kexec, suspend/resume, AER recovery windows) can
 * drop an MSI-X message on the floor. NVIDIA GPUs (Turing and later)
 * track "interrupt already sent" in an on-GPU aggregator: if the message
 * is lost, the corresponding INTR_LEAF pending bit stays latched forever,
 * the host ISR never runs, and completions on that vector go silent.
 *
 * The user-visible symptom is nasty: long (100ms+) kernel-dispatch stalls
 * and degraded workloads with ZERO errors anywhere - no Xid, clean dmesg,
 * clean ECC, all standard health checks green.
 *
 * Detection rule
 * --------------
 * On a healthy GPU a pending bit is cleared by the ISR within
 * microseconds..milliseconds. A bit that stays pending across several
 * reads spread over seconds, while its LEAF_EN_SET enable bit is set and
 * the PCIe MSI-X table/PBA are clean, is a lost interrupt.
 *
 * Register layout (Turing and later; validated on Hopper H100/H200)
 * -----------------------------------------------------------------
 * Source: NVIDIA open-gpu-kernel-modules
 *   src/common/inc/swref/published/hopper/gh100/dev_vm.h
 *   src/nvidia/src/kernel/gpu/intr/arch/turing/intr_cpu_tu102.c
 *
 *   VF register block base in BAR0: 0xb80000
 *     LEAF(i)          = +0x1000 + i*4   per-leaf pending vectors
 *     LEAF_EN_SET(i)   = +0x1200 + i*4   per-leaf enable mask
 *     LEAF_EN_CLEAR(i) = +0x1400 + i*4   write-1-to-clear mirror
 *     TOP              = +0x1600         top-level pending subtree bitmap
 *     TOP_EN_SET       = +0x1608
 *     MSIX_TABLE       = +0x10000
 *     MSIX_PBA         = +0x20000
 *
 * Hopper subtree-to-leaf mapping (each subtree owns 2 leaves):
 *   leaves 0,1   ESCHED_DRIVEN_ENGINE_NOTIFICATION (NONSTALL)
 *   leaves 2,3   UVM_OWNED
 *   leaves 4,5   UVM_SHARED
 *   leaves 6,7   ESCHED_DRIVEN_ENGINE (STALL)
 *   leaves 8,9   RUNLIST + STALL_LAST_SWRL
 *   leaves 10,11 RUNLIST_NOTIFICATION + STALL_LAST
 *
 * Usage:
 *   nv_intr_dump                     # all NVIDIA GPUs, full dump + verdict
 *   nv_intr_dump 0000:53:00.0        # one GPU
 *   nv_intr_dump --repeat 5 --interval-ms 2000
 *   nv_intr_dump --quiet             # verdict lines only
 *
 * Strictly read-only. Requires root (BAR0 access via sysfs resource0).
 *
 * Exit codes:
 *   0 = all scanned GPUs healthy
 *   1 = at least one stuck (latched) enabled pending bit found
 *   2 = usage / IO error
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define VF_BASE          0x00b80000UL
#define VF_OFF_LEAF      0x00001000UL
#define VF_OFF_LEAF_ENS  0x00001200UL
#define VF_OFF_TOP       0x00001600UL
#define VF_OFF_TOP_ENS   0x00001608UL
#define VF_OFF_MSIX      0x00010000UL
#define VF_OFF_MSIX_PBA  0x00020000UL

#define INTR_NUM_LEAVES   16
#define MSIX_NUM_VECTORS  9   /* vectors used on Hopper; others read as spares */
#define MAX_GPUS          16

/* PMC_BOOT_0 top byte = architecture id. The VF-block layout above
 * exists on Turing+; this tool is validated on Hopper. */
static const struct { uint8_t id; const char *name; } known_archs[] = {
    { 0x16, "Turing"  },
    { 0x17, "Ampere"  },
    { 0x18, "Hopper"  },
    { 0x19, "Ada"     },
};

static const char *leaf_role(int i) {
    switch (i) {
    case 0: case 1:   return "ENGINE_NOTIFICATION (NONSTALL)";
    case 2: case 3:   return "UVM_OWNED";
    case 4: case 5:   return "UVM_SHARED";
    case 6: case 7:   return "ENGINE (STALL)";
    case 8: case 9:   return "RUNLIST + STALL_LAST_SWRL";
    case 10: case 11: return "RUNLIST_NOTIFICATION + STALL_LAST";
    default:          return "unused";
    }
}

struct opts {
    int repeat;        /* persistence reads */
    int interval_ms;   /* delay between reads */
    int quiet;
    int no_msix;
    int force;         /* skip arch check */
};

struct gpu_state {
    uint32_t pmc_boot_0;
    uint32_t top, top_ens;
    uint32_t leaf[INTR_NUM_LEAVES];
    uint32_t leaf_ens[INTR_NUM_LEAVES];
    uint64_t pba;
    int      pba_valid;
};

static void *map_region(int fd, unsigned long off, size_t len,
                        unsigned long *within_out, size_t *map_len_out) {
    long page = sysconf(_SC_PAGESIZE);
    unsigned long aligned = off & ~(unsigned long)(page - 1);
    unsigned long within = off - aligned;
    size_t need = within + len;
    size_t map_len = ((need + page - 1) / page) * page;
    void *map = mmap(NULL, map_len, PROT_READ, MAP_SHARED, fd, aligned);
    if (map == MAP_FAILED) return NULL;
    *within_out = within;
    *map_len_out = map_len;
    return map;
}

static int read_u32(int fd, unsigned long off, uint32_t *out) {
    unsigned long within; size_t maplen;
    void *m = map_region(fd, off, 4, &within, &maplen);
    if (!m) return -1;
    *out = *(volatile uint32_t *)((char *)m + within);
    munmap(m, maplen);
    return 0;
}

static int read_state(int fd, struct gpu_state *s) {
    unsigned long within; size_t maplen;
    unsigned long start = VF_BASE + VF_OFF_LEAF;
    size_t span = (VF_OFF_TOP_ENS + 8) - VF_OFF_LEAF;
    void *m = map_region(fd, start, span, &within, &maplen);
    if (!m) return -1;
    char *base = (char *)m + within;
    volatile uint32_t *leaf     = (volatile uint32_t *)(base + 0x000);
    volatile uint32_t *leaf_ens = (volatile uint32_t *)(base + 0x200);
    volatile uint32_t *top      = (volatile uint32_t *)(base + 0x600);
    volatile uint32_t *top_ens  = (volatile uint32_t *)(base + 0x608);
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        s->leaf[i]     = leaf[i];
        s->leaf_ens[i] = leaf_ens[i];
    }
    s->top     = top[0];
    s->top_ens = top_ens[0];
    munmap(m, maplen);

    void *pm = map_region(fd, VF_BASE + VF_OFF_MSIX_PBA, 8, &within, &maplen);
    if (pm) {
        s->pba = *(volatile uint64_t *)((char *)pm + within);
        s->pba_valid = 1;
        munmap(pm, maplen);
    } else {
        s->pba = 0;
        s->pba_valid = 0;
    }
    return 0;
}

static void dump_msix(int fd) {
    unsigned long within; size_t maplen;
    unsigned long msix_off = VF_BASE + VF_OFF_MSIX;
    void *tab = map_region(fd, msix_off, (size_t)MSIX_NUM_VECTORS * 16,
                           &within, &maplen);
    if (!tab) {
        printf("# MSI-X table mmap FAILED at 0x%lx: %s\n",
               msix_off, strerror(errno));
        return;
    }
    volatile uint32_t *t = (volatile uint32_t *)((char *)tab + within);
    unsigned long pwithin; size_t pmaplen;
    void *pm = map_region(fd, VF_BASE + VF_OFF_MSIX_PBA, 8, &pwithin, &pmaplen);
    uint64_t pba = pm ? *(volatile uint64_t *)((char *)pm + pwithin) : 0;

    printf("# === MSI-X table (PCIe spec) @ BAR0+0x%lx ===\n", msix_off);
    printf("# vec  addr_lo     addr_hi     data        vec_ctl     mask  pba\n");
    for (int i = 0; i < MSIX_NUM_VECTORS; i++) {
        uint32_t alo = t[i*4+0], ahi = t[i*4+1], dat = t[i*4+2], vc = t[i*4+3];
        printf("vec%-2d  0x%08x  0x%08x  0x%08x  0x%08x  %u     %d\n",
               i, alo, ahi, dat, vc, vc & 1u,
               pm ? (int)((pba >> i) & 1ULL) : -1);
    }
    if (pm) munmap(pm, pmaplen);
    munmap(tab, maplen);
}

/* Returns: 0 healthy, 1 stuck, 2 error. */
static int check_gpu(const char *bdf, const struct opts *o) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource0", bdf);
    int fd = open(path, O_RDONLY | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "ERROR open(%s): %s\n", path, strerror(errno));
        return 2;
    }

    uint32_t pmc = 0;
    if (read_u32(fd, 0x0, &pmc) != 0 || pmc == 0 || pmc == 0xFFFFFFFF) {
        fprintf(stderr, "ERROR %s: PMC_BOOT_0 reads 0x%08x - BAR0 MMIO broken\n",
                bdf, pmc);
        close(fd);
        return 2;
    }
    uint8_t arch = (pmc >> 24) & 0xFF;
    const char *arch_name = NULL;
    for (size_t i = 0; i < sizeof(known_archs)/sizeof(known_archs[0]); i++)
        if (known_archs[i].id == arch) arch_name = known_archs[i].name;
    if (!arch_name && !o->force) {
        fprintf(stderr, "SKIP %s: PMC_BOOT_0=0x%08x arch 0x%02x not in known "
                "list (use --force to scan anyway)\n", bdf, pmc, arch);
        close(fd);
        return 2;
    }

    /* Persistence sampling: stuck = AND of pending across all reads. */
    struct gpu_state first, cur;
    uint32_t stuck[INTR_NUM_LEAVES];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (read_state(fd, &first) != 0) {
        fprintf(stderr, "ERROR %s: mmap aggregator: %s\n", bdf, strerror(errno));
        close(fd);
        return 2;
    }
    memcpy(stuck, first.leaf, sizeof(stuck));
    for (int r = 1; r < o->repeat; r++) {
        struct timespec ts = { o->interval_ms / 1000,
                               (o->interval_ms % 1000) * 1000000L };
        nanosleep(&ts, NULL);
        if (read_state(fd, &cur) != 0) { close(fd); return 2; }
        for (int i = 0; i < INTR_NUM_LEAVES; i++)
            stuck[i] &= cur.leaf[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long span_ms = (t1.tv_sec - t0.tv_sec) * 1000
                 + (t1.tv_nsec - t0.tv_nsec) / 1000000;

    if (!o->quiet) {
        printf("# bdf=%s  PMC_BOOT_0=0x%08x  arch=%s\n",
               bdf, pmc, arch_name ? arch_name : "unknown(forced)");
        if (!o->no_msix) dump_msix(fd);
        printf("# === Interrupt aggregator (first read) ===\n");
        printf("TOP        0x%08x   # pending subtrees\n", first.top);
        printf("TOP_ENSET  0x%08x\n", first.top_ens);
        printf("# leaf  pending     en_set      stuck(x%d/%ldms)  role\n",
               o->repeat, span_ms);
        for (int i = 0; i < INTR_NUM_LEAVES; i++) {
            printf("leaf%-2d  0x%08x  0x%08x  0x%08x       %s\n",
                   i, first.leaf[i], first.leaf_ens[i], stuck[i],
                   leaf_role(i));
        }
    }

    /* Verdict: any stuck bit that is also ENABLED is a lost interrupt.
     * Stuck-but-disabled bits are reported informationally: they cannot
     * fire an MSI-X, but a persistently pending disabled bit is still
     * unusual. */
    int stuck_enabled = 0, stuck_disabled = 0;
    for (int i = 0; i < INTR_NUM_LEAVES; i++) {
        uint32_t en  = stuck[i] & first.leaf_ens[i];
        uint32_t dis = stuck[i] & ~first.leaf_ens[i];
        if (en)  stuck_enabled++;
        if (dis) stuck_disabled++;
        if (en)
            printf("VERDICT %s STUCK leaf=%d bits=0x%08x persisted=%d/%d span_ms=%ld role=\"%s\"\n",
                   bdf, i, en, o->repeat, o->repeat, span_ms, leaf_role(i));
        if (dis && !o->quiet)
            printf("INFO %s persistent-but-disabled leaf=%d bits=0x%08x\n",
                   bdf, i, dis);
    }
    if (!stuck_enabled)
        printf("VERDICT %s OK\n", bdf);

    close(fd);
    return stuck_enabled ? 1 : 0;
}

static int discover_nvidia_gpus(char bdfs[][16], int max) {
    DIR *d = opendir("/sys/bus/pci/devices");
    if (!d) return 0;
    int n = 0;
    struct dirent *de;
    while ((de = readdir(d)) && n < max) {
        if (de->d_name[0] == '.') continue;
        if (strlen(de->d_name) != 12) continue;  /* dddd:bb:dd.f */
        char vp[300];
        snprintf(vp, sizeof(vp), "/sys/bus/pci/devices/%s/vendor", de->d_name);
        FILE *fp = fopen(vp, "r");
        if (!fp) continue;
        char vbuf[16] = {0};
        if (fgets(vbuf, sizeof(vbuf), fp) &&
            strncasecmp(vbuf, "0x10de", 6) == 0)
            snprintf(bdfs[n++], 16, "%s", de->d_name);
        fclose(fp);
    }
    closedir(d);
    /* sort for stable output */
    for (int i = 0; i + 1 < n; i++)
        for (int j = i + 1; j < n; j++)
            if (strcmp(bdfs[i], bdfs[j]) > 0) {
                char tmp[16];
                memcpy(tmp, bdfs[i], 16);
                memcpy(bdfs[i], bdfs[j], 16);
                memcpy(bdfs[j], tmp, 16);
            }
    return n;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options] [bdf ...]\n"
        "  bdf                 PCI address, e.g. 0000:53:00.0 (default: all NVIDIA GPUs)\n"
        "  --repeat N          persistence reads per GPU (default 3)\n"
        "  --interval-ms M     delay between reads (default 1000)\n"
        "  --quiet             print only VERDICT lines\n"
        "  --no-msix           skip MSI-X table dump\n"
        "  --force             scan unrecognized GPU architectures too\n"
        "exit: 0 healthy, 1 stuck bits found, 2 error\n", prog);
}

int main(int argc, char **argv) {
    struct opts o = { .repeat = 3, .interval_ms = 1000 };
    char bdfs[MAX_GPUS][16];
    int n_bdf = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--repeat")      && i+1 < argc) o.repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--interval-ms") && i+1 < argc) o.interval_ms = atoi(argv[++i]);
        else if (!strcmp(a, "--quiet"))   o.quiet = 1;
        else if (!strcmp(a, "--no-msix")) o.no_msix = 1;
        else if (!strcmp(a, "--force"))   o.force = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else if (a[0] == '-') { usage(argv[0]); return 2; }
        else if (n_bdf < MAX_GPUS) snprintf(bdfs[n_bdf++], 16, "%s", a);
    }
    if (o.repeat < 1) o.repeat = 1;

    if (n_bdf == 0) {
        n_bdf = discover_nvidia_gpus(bdfs, MAX_GPUS);
        if (n_bdf == 0) {
            fprintf(stderr, "ERROR: no NVIDIA GPUs found in /sys/bus/pci/devices\n");
            return 2;
        }
    }

    int any_stuck = 0, any_err = 0;
    for (int i = 0; i < n_bdf; i++) {
        int rc = check_gpu(bdfs[i], &o);
        if (rc == 1) any_stuck = 1;
        if (rc == 2) any_err = 1;
        if (!o.quiet && i + 1 < n_bdf) printf("\n");
    }
    if (any_stuck) return 1;
    return any_err ? 2 : 0;
}
