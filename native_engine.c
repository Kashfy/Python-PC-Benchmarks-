/*
 * native_engine.c — portable native benchmark engine (CPU int, CPU float,
 * memory copy, disk I/O). Compiler-optimized counterpart to benchmark.py.
 *
 * Builds cleanly on Windows (MSVC / MinGW), macOS, and Linux.
 *
 * Build:
 *   POSIX : cc -O2 native_engine.c -o native_engine -lm
 *   MinGW : gcc -O2 native_engine.c -o native_engine.exe
 *   MSVC  : cl /O2 native_engine.c
 *
 * Run:
 *   ./native_engine                       # human-readable
 *   ./native_engine --json                # machine-readable JSON (used by
 *                                          # benchmark.py)
 *   ./native_engine --seconds 5 --repeats 5
 *
 * Reported units match benchmark.py so numbers are comparable:
 *   CPU Integer -> primes/s, CPU Float -> iters/s,
 *   Memory      -> MB/s copy, Disk write/read -> MB/s.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --------------------------- Portable timing --------------------------- */
#if defined(_WIN32)
  #include <windows.h>
  static double now_seconds(void) {
      static LARGE_INTEGER freq;
      static int inited = 0;
      LARGE_INTEGER c;
      if (!inited) { QueryPerformanceFrequency(&freq); inited = 1; }
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart / (double)freq.QuadPart;
  }
#else
  #include <time.h>
  static double now_seconds(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
  }
#endif

/* --------------------------- Portable temp file ------------------------ */
#if defined(_WIN32)
  #include <io.h>
  #include <windows.h>
#else
  #include <unistd.h>
  #include <fcntl.h>
#endif

#define MB (1024 * 1024)
#define PRIME_LO 50000
#define PRIME_HI 51000
#define PRIMES_PER_CHUNK (PRIME_HI - PRIME_LO)
#define FLOAT_ITERS_PER_CHUNK 50000

/* --------------------------- Workloads --------------------------------- */
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    int r = (int)sqrt((double)n);
    for (int i = 3; i <= r; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

/* volatile sink so the optimizer keeps the work but we stay honest */
static volatile long g_sink_i = 0;
static volatile double g_sink_d = 0.0;

static void cpu_integer_chunk(void) {
    long acc = 0;
    for (int n = PRIME_LO; n < PRIME_HI; ++n) acc += is_prime(n);
    g_sink_i += acc;
}

static void cpu_float_chunk(void) {
    double x = 0.001, s = 0.0;
    for (int i = 0; i < FLOAT_ITERS_PER_CHUNK; ++i) {
        x += 0.00001;
        s += sin(x) * cos(x) + sqrt(x);
    }
    g_sink_d += s;
}

/* --------------------------- Stats ------------------------------------- */
static double median(double *v, int n) {
    /* simple insertion sort; n is tiny (repeats) */
    for (int i = 1; i < n; ++i) {
        double key = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
        v[j + 1] = key;
    }
    if (n == 0) return 0.0;
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double stddev(const double *v, int n) {
    if (n < 2) return 0.0;
    double m = 0.0;
    for (int i = 0; i < n; ++i) m += v[i];
    m /= n;
    double s = 0.0;
    for (int i = 0; i < n; ++i) { double d = v[i] - m; s += d * d; }
    return sqrt(s / (n - 1));
}

/* --------------------------- Benchmark runners ------------------------- */
typedef void (*chunk_fn)(void);

/* Runs `func` for ~seconds, returns work-units/second (units_per_chunk *
 * chunks / elapsed). */
static double run_rate(chunk_fn func, double seconds, long units_per_chunk) {
    double start = now_seconds(), elapsed = 0.0;
    long long chunks = 0;
    do {
        func();
        ++chunks;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);
    return (double)(chunks * units_per_chunk) / elapsed;
}

/* Memory copy bandwidth (MB/s) via memcpy of a large buffer. */
static double run_memory(double seconds, int buf_mb) {
    size_t n = (size_t)buf_mb * MB;
    char *src = (char *)malloc(n);
    char *dst = (char *)malloc(n);
    if (!src || !dst) { free(src); free(dst); return 0.0; }
    memset(src, 'A', n);
    double start = now_seconds(), elapsed = 0.0;
    unsigned long long copied = 0;
    do {
        memcpy(dst, src, n);
        copied += n;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);
    g_sink_i += dst[0];
    free(src); free(dst);
    return (double)copied / elapsed / (double)MB;
}

/* Disk sequential write & read (MB/s). Writes via a temp file. */
static void run_disk(double seconds, int file_mb,
                     double *out_write, double *out_read) {
    (void)seconds;
    size_t chunk = 4 * (size_t)MB;
    long n_chunks = (long)(((size_t)file_mb * MB) / chunk);
    if (n_chunks < 1) n_chunks = 1;
    size_t total = (size_t)n_chunks * chunk;
    char *buf = (char *)malloc(chunk);
    if (!buf) { *out_write = *out_read = 0.0; return; }
    memset(buf, 'X', chunk);

    char path[512];
#if defined(_WIN32)
    char tmpdir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmpdir);
    snprintf(path, sizeof(path), "%snative_bench_%lu.bin",
             tmpdir, (unsigned long)GetCurrentProcessId());
    FILE *f = fopen(path, "wb");
    if (!f) { free(buf); *out_write = *out_read = 0.0; return; }
    double t0 = now_seconds();
    for (long i = 0; i < n_chunks; ++i) fwrite(buf, 1, chunk, f);
    fflush(f);
    /* best-effort flush to disk on Windows */
    fclose(f);
    *out_write = (double)total / (now_seconds() - t0) / (double)MB;

    f = fopen(path, "rb");
    if (!f) { free(buf); remove(path); *out_read = 0.0; return; }
    t0 = now_seconds();
    size_t got = 0, r;
    while ((r = fread(buf, 1, chunk, f)) > 0) got += r;
    fclose(f);
    *out_read = (double)got / (now_seconds() - t0) / (double)MB;
    remove(path);
#else
    strcpy(path, "/tmp/native_bench_XXXXXX");
    int fd = mkstemp(path);
    if (fd == -1) { free(buf); *out_write = *out_read = 0.0; return; }
    double t0 = now_seconds();
    for (long i = 0; i < n_chunks; ++i) {
        if (write(fd, buf, chunk) < 0) break;
    }
    fsync(fd);
    *out_write = (double)total / (now_seconds() - t0) / (double)MB;

    lseek(fd, 0, SEEK_SET);
    t0 = now_seconds();
    size_t got = 0; ssize_t r;
    while ((r = read(fd, buf, chunk)) > 0) got += (size_t)r;
    *out_read = (double)got / (now_seconds() - t0) / (double)MB;
    close(fd);
    unlink(path);
#endif
    g_sink_i += buf[0];
    free(buf);
}

/* --------------------------- Output ------------------------------------ */
typedef struct { const char *name; const char *unit; double rate, stdev; } Res;

static void print_human(Res *r, int n) {
    printf("\n=== Native (C) Engine ===\n");
    for (int i = 0; i < n; ++i)
        printf("  %-24s: %14.2f %-8s (stdev %.2f)\n",
               r[i].name, r[i].rate, r[i].unit, r[i].stdev);
}

static void print_json(Res *r, int n, double seconds, int repeats) {
    printf("{\n");
    printf("  \"engine\": \"native-c\",\n");
    printf("  \"seconds\": %.3f,\n", seconds);
    printf("  \"repeats\": %d,\n", repeats);
    printf("  \"results\": [\n");
    for (int i = 0; i < n; ++i) {
        printf("    {\"name\": \"%s\", \"unit\": \"%s\", \"rate\": %.4f, "
               "\"stdev\": %.4f}%s\n",
               r[i].name, r[i].unit, r[i].rate, r[i].stdev,
               (i == n - 1) ? "" : ",");
    }
    printf("  ]\n}\n");
}

/* --------------------------- Main -------------------------------------- */
int main(int argc, char **argv) {
    double seconds = 3.0;
    int repeats = 3, as_json = 0;
    int mem_mb = 64, disk_mb = 256;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json")) as_json = 1;
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc)
            repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-mb") && i + 1 < argc)
            mem_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--disk-mb") && i + 1 < argc)
            disk_mb = atoi(argv[++i]);
        else {
            fprintf(stderr,
                "Usage: %s [--json] [--seconds N] [--repeats M] "
                "[--mem-mb K] [--disk-mb K]\n", argv[0]);
            return 1;
        }
    }
    if (repeats < 1) repeats = 1;

    double *ci = malloc(sizeof(double) * repeats);
    double *cf = malloc(sizeof(double) * repeats);
    double *mm = malloc(sizeof(double) * repeats);
    double *dw = malloc(sizeof(double) * repeats);
    double *dr = malloc(sizeof(double) * repeats);
    if (!ci || !cf || !mm || !dw || !dr) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (int r = 0; r < repeats; ++r) {
        ci[r] = run_rate(cpu_integer_chunk, seconds, PRIMES_PER_CHUNK);
        cf[r] = run_rate(cpu_float_chunk, seconds, FLOAT_ITERS_PER_CHUNK);
        mm[r] = run_memory(seconds, mem_mb);
        run_disk(seconds, disk_mb, &dw[r], &dr[r]);
    }

    Res results[5];
    results[0] = (Res){"CPU Integer (primes)", "primes/s",
                       median(ci, repeats), stddev(ci, repeats)};
    results[1] = (Res){"CPU Float (math ops)", "iters/s",
                       median(cf, repeats), stddev(cf, repeats)};
    results[2] = (Res){"Memory copy bandwidth", "MB/s",
                       median(mm, repeats), stddev(mm, repeats)};
    results[3] = (Res){"Disk write", "MB/s",
                       median(dw, repeats), stddev(dw, repeats)};
    results[4] = (Res){"Disk read", "MB/s",
                       median(dr, repeats), stddev(dr, repeats)};

    if (as_json) print_json(results, 5, seconds, repeats);
    else         print_human(results, 5);

    free(ci); free(cf); free(mm); free(dw); free(dr);
    /* touch sinks so they are not optimized away */
    if (g_sink_i == 0xDEADBEEF && g_sink_d == 1.5) fputs("", stderr);
    return 0;
}
