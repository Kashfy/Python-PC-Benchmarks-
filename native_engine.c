/*
 * native_engine.c — portable native benchmark engine.
 *
 * Compiler-optimized counterpart to the Python benchmarks, plus two
 * measurements Python cannot express meaningfully:
 *
 *   - multi-threaded CPU scaling (real threads, no GIL, no process overhead)
 *   - memory latency by working-set size (pointer chase), which resolves the
 *     L1/L2/L3/DRAM hierarchy; in Python the interpreter overhead per access
 *     is an order of magnitude larger than an L1 hit, so the signal is lost.
 *
 * Builds on Windows (MSVC / MinGW), macOS, and Linux.
 *
 *   POSIX : cc -O2 native_engine.c -o native_engine -lm -lpthread
 *   MinGW : gcc -O2 native_engine.c -o native_engine.exe
 *   MSVC  : cl /O2 native_engine.c
 *
 * Run:
 *   ./native_engine                  # human-readable
 *   ./native_engine --json           # machine-readable (used by pcbench)
 *   ./native_engine --seconds 5 --repeats 5 --threads 8
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --------------------------- Platform layer ---------------------------- */
#if defined(_WIN32)
  #include <windows.h>
  #include <io.h>
  typedef HANDLE thread_t;
  static double now_seconds(void) {
      static LARGE_INTEGER freq; static int inited = 0;
      LARGE_INTEGER c;
      if (!inited) { QueryPerformanceFrequency(&freq); inited = 1; }
      QueryPerformanceCounter(&c);
      return (double)c.QuadPart / (double)freq.QuadPart;
  }
  static int cpu_count(void) {
      SYSTEM_INFO si; GetSystemInfo(&si);
      return (int)si.dwNumberOfProcessors;
  }
#else
  #include <time.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <pthread.h>
  typedef pthread_t thread_t;
  static double now_seconds(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
  }
  static int cpu_count(void) {
      long n = sysconf(_SC_NPROCESSORS_ONLN);
      return n > 0 ? (int)n : 1;
  }
#endif

#define MB (1024 * 1024)
#define KB 1024
#define PRIME_LO 50000
#define PRIME_HI 51000
#define PRIMES_PER_CHUNK (PRIME_HI - PRIME_LO)
#define EXPECTED_PRIMES 89
#define FLOAT_ITERS_PER_CHUNK 50000

/* Volatile sinks keep the optimizer from deleting the work outright while
 * still allowing it to optimize the work itself. */
static volatile long long g_sink_i = 0;
static volatile double    g_sink_d = 0.0;
static int g_validation_failed = 0;

/* --------------------------- Workloads --------------------------------- */
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    int r = (int)sqrt((double)n);
    for (int i = 3; i <= r; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

/* Returns the prime count so callers can verify the machine computed the
 * right answer — a wrong result means unstable hardware, not a fast machine. */
static int cpu_integer_chunk(void) {
    int count = 0;
    for (int n = PRIME_LO; n < PRIME_HI; ++n) count += is_prime(n);
    g_sink_i += count;
    return count;
}

static double cpu_float_chunk(void) {
    double x = 0.001, s = 0.0;
    for (int i = 0; i < FLOAT_ITERS_PER_CHUNK; ++i) {
        x += 0.00001;
        s += sin(x) * cos(x) + sqrt(x);
    }
    g_sink_d += s;
    return s;
}

/* --------------------------- Statistics -------------------------------- */
static double median(double *v, int n) {
    for (int i = 1; i < n; ++i) {          /* insertion sort; n is tiny */
        double key = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
        v[j + 1] = key;
    }
    if (n <= 0) return 0.0;
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

/* --------------------------- Single-thread runners --------------------- */
static double run_rate(double seconds, long units_per_chunk, int is_float) {
    double start = now_seconds(), elapsed;
    long long chunks = 0;
    do {
        if (is_float) cpu_float_chunk();
        else if (cpu_integer_chunk() != EXPECTED_PRIMES) g_validation_failed = 1;
        ++chunks;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);
    return (double)(chunks * units_per_chunk) / elapsed;
}

static double run_memory(double seconds, int buf_mb) {
    size_t n = (size_t)buf_mb * MB;
    char *src = (char *)malloc(n), *dst = (char *)malloc(n);
    if (!src || !dst) { free(src); free(dst); return 0.0; }
    memset(src, 'A', n);
    double start = now_seconds(), elapsed;
    unsigned long long copied = 0;
    do {
        memcpy(dst, src, n);
        copied += n;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);
    if (memcmp(src, dst, 4096) != 0) g_validation_failed = 1;
    g_sink_i += dst[0];
    free(src); free(dst);
    return (double)copied / elapsed / (double)MB;
}

/* --------------------------- Multi-threaded CPU ------------------------ */
typedef struct {
    double seconds;
    long long primes;   /* out */
    int ok;             /* out: validation */
} worker_arg_t;

#if defined(_WIN32)
static DWORD WINAPI cpu_worker(LPVOID p)
#else
static void *cpu_worker(void *p)
#endif
{
    worker_arg_t *a = (worker_arg_t *)p;
    double start = now_seconds();
    long long chunks = 0;
    a->ok = 1;
    while (now_seconds() - start < a->seconds) {
        if (cpu_integer_chunk() != EXPECTED_PRIMES) a->ok = 0;
        ++chunks;
    }
    a->primes = chunks * PRIMES_PER_CHUNK;
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

/* Aggregate primes/s across `nthreads` real threads. Unlike the Python
 * multiprocessing path this has no interpreter, no GIL, and no process
 * spawn cost, so it shows the hardware's true parallel ceiling. */
static double run_multithread(double seconds, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    worker_arg_t *args = (worker_arg_t *)calloc(nthreads, sizeof(*args));
    thread_t *threads = (thread_t *)calloc(nthreads, sizeof(*threads));
    if (!args || !threads) { free(args); free(threads); return 0.0; }

    double start = now_seconds();
    for (int i = 0; i < nthreads; ++i) {
        args[i].seconds = seconds;
#if defined(_WIN32)
        threads[i] = CreateThread(NULL, 0, cpu_worker, &args[i], 0, NULL);
#else
        if (pthread_create(&threads[i], NULL, cpu_worker, &args[i]) != 0)
            threads[i] = 0;
#endif
    }
    long long total = 0;
    for (int i = 0; i < nthreads; ++i) {
#if defined(_WIN32)
        if (threads[i]) { WaitForSingleObject(threads[i], INFINITE);
                          CloseHandle(threads[i]); }
#else
        if (threads[i]) pthread_join(threads[i], NULL);
#endif
        total += args[i].primes;
        if (!args[i].ok) g_validation_failed = 1;
    }
    double wall = now_seconds() - start;
    free(args); free(threads);
    return wall > 0 ? (double)total / wall : 0.0;
}

/* --------------------------- Memory latency ---------------------------- */
/* Sattolo's algorithm builds a permutation that is a single cycle, so the
 * chase visits every slot exactly once before repeating and the CPU's
 * prefetcher cannot predict the next address. */
static void build_cycle(size_t *arr, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) arr[i] = i;
    unsigned long long rng = seed ? seed : 1;
    for (size_t i = n - 1; i > 0; --i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (size_t)((rng >> 33) % i);      /* j < i => single cycle */
        size_t t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* Average nanoseconds per dependent load for a given working-set size. */
static double pointer_chase_ns(size_t bytes, double seconds) {
    size_t n = bytes / sizeof(size_t);
    if (n < 2) return 0.0;
    size_t *arr = (size_t *)malloc(n * sizeof(size_t));
    if (!arr) return 0.0;
    build_cycle(arr, n, 12345u);

    size_t idx = 0;
    for (size_t i = 0; i < n; ++i) idx = arr[idx];   /* warm the caches */

    const int BATCH = 1024;
    double start = now_seconds(), elapsed;
    long long steps = 0;
    do {
        for (int i = 0; i < BATCH; ++i) idx = arr[idx];
        steps += BATCH;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);

    g_sink_i += (long long)idx;   /* keep the chase alive */
    free(arr);
    return steps ? elapsed / (double)steps * 1e9 : 0.0;
}

/* --------------------------- Disk -------------------------------------- */
static void run_disk(int file_mb, double *out_write, double *out_read) {
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
  #if defined(__APPLE__)
    fcntl(fd, 48 /* F_NOCACHE */, 1);   /* before writing: keep it uncached */
  #endif
    double t0 = now_seconds();
    for (long i = 0; i < n_chunks; ++i)
        if (write(fd, buf, chunk) < 0) break;
    fsync(fd);
    *out_write = (double)total / (now_seconds() - t0) / (double)MB;

  #if defined(__linux__)
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  #endif
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
typedef struct { const char *name; const char *unit; double rate, sd; } Res;

static const size_t LAT_SIZES[] = {
    16 * (size_t)KB, 64 * (size_t)KB, 256 * (size_t)KB, 1 * (size_t)MB,
    4 * (size_t)MB, 16 * (size_t)MB, 64 * (size_t)MB, 256 * (size_t)MB
};
#define N_LAT (sizeof(LAT_SIZES) / sizeof(LAT_SIZES[0]))

static void size_label(size_t b, char *out, size_t cap) {
    if (b >= MB) snprintf(out, cap, "%zu MB", b / MB);
    else         snprintf(out, cap, "%zu KB", b / KB);
}

static void print_human(Res *r, int n, double *lat) {
    printf("\n=== Native (C) Engine ===\n");
    for (int i = 0; i < n; ++i)
        printf("  %-26s: %14.2f %-9s (stdev %.2f)\n",
               r[i].name, r[i].rate, r[i].unit, r[i].sd);
    printf("\n  Memory latency (pointer chase):\n");
    for (size_t i = 0; i < N_LAT; ++i) {
        char lbl[32]; size_label(LAT_SIZES[i], lbl, sizeof(lbl));
        printf("    %8s : %7.2f ns\n", lbl, lat[i]);
    }
    if (g_validation_failed)
        printf("\n  !! VALIDATION FAILED — computed results were incorrect\n");
}

static void print_json(Res *r, int n, double *lat,
                       double seconds, int repeats, int threads) {
    printf("{\n  \"engine\": \"native-c\",\n");
    printf("  \"seconds\": %.3f,\n  \"repeats\": %d,\n  \"threads\": %d,\n",
           seconds, repeats, threads);
    printf("  \"validated\": %s,\n", g_validation_failed ? "false" : "true");
    printf("  \"results\": [\n");
    for (int i = 0; i < n; ++i)
        printf("    {\"name\": \"%s\", \"unit\": \"%s\", \"rate\": %.4f, "
               "\"stdev\": %.4f}%s\n",
               r[i].name, r[i].unit, r[i].rate, r[i].sd,
               (i == n - 1) ? "" : ",");
    printf("  ],\n  \"latency\": [\n");
    for (size_t i = 0; i < N_LAT; ++i) {
        char lbl[32]; size_label(LAT_SIZES[i], lbl, sizeof(lbl));
        printf("    {\"label\": \"%s\", \"bytes\": %zu, \"ns\": %.3f}%s\n",
               lbl, LAT_SIZES[i], lat[i], (i == N_LAT - 1) ? "" : ",");
    }
    printf("  ]\n}\n");
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--json] [--seconds N] [--repeats M] [--threads T]\n"
            "          [--mem-mb K] [--disk-mb K]\n", prog);
}

/* --------------------------- Main -------------------------------------- */
int main(int argc, char **argv) {
    double seconds = 3.0;
    int repeats = 3, as_json = 0, threads = 0, mem_mb = 64, disk_mb = 256;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json")) as_json = 1;
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-mb")  && i + 1 < argc) mem_mb  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--disk-mb") && i + 1 < argc) disk_mb = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (repeats < 1) repeats = 1;
    if (threads < 1) threads = cpu_count();

    double *ci = (double *)malloc(sizeof(double) * repeats);
    double *cf = (double *)malloc(sizeof(double) * repeats);
    double *mt = (double *)malloc(sizeof(double) * repeats);
    double *mm = (double *)malloc(sizeof(double) * repeats);
    double *dw = (double *)malloc(sizeof(double) * repeats);
    double *dr = (double *)malloc(sizeof(double) * repeats);
    if (!ci || !cf || !mt || !mm || !dw || !dr) {
        fprintf(stderr, "out of memory\n"); return 1;
    }

    for (int r = 0; r < repeats; ++r) {
        ci[r] = run_rate(seconds, PRIMES_PER_CHUNK, 0);
        cf[r] = run_rate(seconds, FLOAT_ITERS_PER_CHUNK, 1);
        mt[r] = run_multithread(seconds, threads);
        mm[r] = run_memory(seconds, mem_mb);
        run_disk(disk_mb, &dw[r], &dr[r]);
    }

    /* Latency is measured once: it is a property of the memory hierarchy and
     * does not benefit from repetition the way throughput does. */
    double lat[N_LAT];
    double lat_budget = seconds / (double)N_LAT;
    if (lat_budget < 0.05) lat_budget = 0.05;
    for (size_t i = 0; i < N_LAT; ++i)
        lat[i] = pointer_chase_ns(LAT_SIZES[i], lat_budget);

    Res results[6];
    results[0] = (Res){"CPU Integer (primes)",  "primes/s", median(ci, repeats), stddev(ci, repeats)};
    results[1] = (Res){"CPU Float (math ops)",  "iters/s",  median(cf, repeats), stddev(cf, repeats)};
    results[2] = (Res){"CPU Multi-thread",      "primes/s", median(mt, repeats), stddev(mt, repeats)};
    results[3] = (Res){"Memory copy bandwidth", "MB/s",     median(mm, repeats), stddev(mm, repeats)};
    results[4] = (Res){"Disk write",            "MB/s",     median(dw, repeats), stddev(dw, repeats)};
    results[5] = (Res){"Disk read",             "MB/s",     median(dr, repeats), stddev(dr, repeats)};

    if (as_json) print_json(results, 6, lat, seconds, repeats, threads);
    else         print_human(results, 6, lat);

    free(ci); free(cf); free(mt); free(mm); free(dw); free(dr);
    if (g_sink_i == 0x7FFFFFFF && g_sink_d == 1.5) fputs("", stderr);
    return g_validation_failed ? 2 : 0;
}
