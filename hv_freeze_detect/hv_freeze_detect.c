/*
 * hv_freeze_detect.c
 * Detect short VM pauses (e.g. hypervisor kexec) by measuring scheduling gaps.
 *
 * Compile : gcc -O2 -Wall -o hv_freeze_detect hv_freeze_detect.c
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

static volatile int stop = 0;
static void on_sig(int s) { (void)s; stop = 1; }

static double diff_ms(struct timespec a, struct timespec b) {
    return (a.tv_sec - b.tv_sec) * 1000.0 +
           (a.tv_nsec - b.tv_nsec) / 1.0e6;
}

/* Format current UTC time into ISO8601 string */
static void current_time_iso8601(char *buf, size_t len) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);
    snprintf(buf, len, "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

int main(int argc, char **argv)
{
    long period_us = 1000;   // default 1 ms
    double threshold_ms = 50.0;

    if (argc > 1) period_us = atol(argv[1]);
    if (argc > 2) threshold_ms = atof(argv[2]);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    struct timespec t_prev, t_now;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t_prev);

    while (!stop) {
        usleep(period_us);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_now);

        double gap_ms = diff_ms(t_now, t_prev);
        if (gap_ms > threshold_ms) {
            char tsbuf[64];
            current_time_iso8601(tsbuf, sizeof(tsbuf));
            printf("{\"ts\":\"%s\",\"event\":\"pause\",\"gap_ms\":%.3f}\n",
                   tsbuf, gap_ms);
            fflush(stdout);
        }

        t_prev = t_now;
    }

    return 0;
}
