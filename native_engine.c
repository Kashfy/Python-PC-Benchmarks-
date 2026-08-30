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

/* --------------------------- Resource probes ---------------------------
 * The engine must not allocate more than the machine has, nor fill its disk.
 * Over-allocating drives the OS into swap (stalling the machine and writing
 * heavily to the SSD); filling the disk can corrupt other programs' writes.
 * On many Linux systems /tmp is tmpfs — RAM — so the disk test must check
 * free space there rather than assume it is backed by storage. */
#if defined(_WIN32)
static unsigned long long total_ram_bytes(void) {
    MEMORYSTATUSEX st; st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (unsigned long long)st.ullTotalPhys : 0;
}
static unsigned long long free_disk_bytes(const char *path) {
    ULARGE_INTEGER avail;
    if (GetDiskFreeSpaceExA(path, &avail, NULL, NULL))
        return (unsigned long long)avail.QuadPart;
    return 0;
}
#else
  #include <sys/statvfs.h>
static unsigned long long total_ram_bytes(void) {
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0)
        return (unsigned long long)pages * (unsigned long long)page;
    return 0;
}
static unsigned long long free_disk_bytes(const char *path) {
    struct statvfs vfs;
    if (statvfs(path, &vfs) == 0)
        return (unsigned long long)vfs.f_bavail *
               (unsigned long long)vfs.f_frsize;
    return 0;
}
#endif

/* Never let a single allocation exceed this share of physical RAM. */
#define RAM_SAFE_FRACTION 8          /* i.e. 1/8 of total RAM */
/* Require this much more free space than the file we intend to write. */
#define DISK_HEADROOM 1.5

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

/* --------------------------- STREAM ------------------------------------
 * The industry-standard memory-bandwidth benchmark (John McCalpin, 1991).
 * Four kernels over three double arrays:
 *
 *   Copy   c[j] = a[j]              2 arrays touched
 *   Scale  b[j] = q * c[j]          2 arrays touched
 *   Add    c[j] = a[j] + b[j]       3 arrays touched
 *   Triad  a[j] = b[j] + q * c[j]   3 arrays touched
 *
 * Triad is the figure people quote, because it is the one with a
 * read-read-write pattern and an arithmetic operation, which is what real
 * numerical code looks like.
 *
 * Two rules from the reference implementation are honoured here because
 * violating either silently invalidates the number:
 *
 *   1. Each array must be considerably larger than the last-level cache,
 *      or the benchmark measures cache bandwidth instead of memory bandwidth.
 *      STREAM requires 4x LLC; the array size is chosen from RAM below and
 *      reported so the caller can judge.
 *   2. The result must be validated. A compiler that hoists or vectorises
 *      away the loop produces a spectacular and meaningless number, so the
 *      final array contents are checked against the arithmetic they should
 *      have produced.
 *
 * STREAM reports MB/s in powers of ten (1 MB = 1e6 bytes), not powers of two.
 * That convention is kept so figures are directly comparable to published
 * STREAM results; it is why these numbers look ~5% higher than a MiB/s figure
 * for the same hardware.
 */
#define STREAM_SCALAR 3.0

typedef struct {
    double copy, scale, add, triad;
    size_t elements;
    size_t array_bytes;
    int validated;
} StreamResult;

static int stream_validate(const double *a, const double *b, const double *c,
                           size_t n, int ntimes) {
    /* Replay the kernels on scalars: every element underwent identical
     * arithmetic, so one scalar trace predicts the whole array. */
    double aj = 1.0, bj = 2.0, cj = 0.0, q = STREAM_SCALAR;
    for (int k = 0; k < ntimes; ++k) {
        cj = aj;              /* Copy  */
        bj = q * cj;          /* Scale */
        cj = aj + bj;         /* Add   */
        aj = bj + q * cj;     /* Triad */
    }
    double ea = 0.0, eb = 0.0, ec = 0.0;
    for (size_t j = 0; j < n; ++j) {
        ea += fabs(a[j] - aj);
        eb += fabs(b[j] - bj);
        ec += fabs(c[j] - cj);
    }
    double tol = 1e-8;
    return (ea / n < tol * fabs(aj) + tol)
        && (eb / n < tol * fabs(bj) + tol)
        && (ec / n < tol * fabs(cj) + tol);
}

static int run_stream(StreamResult *out, size_t array_bytes, int ntimes) {
    size_t n = array_bytes / sizeof(double);
    if (n < 1024) return 0;

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));
    if (!a || !b || !c) { free(a); free(b); free(c); return 0; }

    for (size_t j = 0; j < n; ++j) { a[j] = 1.0; b[j] = 2.0; c[j] = 0.0; }
    if (ntimes < 2) ntimes = 2;

    /* Best (not mean) time per kernel, as the reference implementation does:
     * the fastest pass is the one least disturbed by interference, and the
     * quantity being characterised is the hardware's capability. */
    double best_copy = 1e30, best_scale = 1e30;
    double best_add = 1e30, best_triad = 1e30;
    double q = STREAM_SCALAR, t;

    for (int k = 0; k < ntimes; ++k) {
        t = now_seconds();
        for (size_t j = 0; j < n; ++j) c[j] = a[j];
        t = now_seconds() - t;
        if (k > 0 && t < best_copy) best_copy = t;

        t = now_seconds();
        for (size_t j = 0; j < n; ++j) b[j] = q * c[j];
        t = now_seconds() - t;
        if (k > 0 && t < best_scale) best_scale = t;

        t = now_seconds();
        for (size_t j = 0; j < n; ++j) c[j] = a[j] + b[j];
        t = now_seconds() - t;
        if (k > 0 && t < best_add) best_add = t;

        t = now_seconds();
        for (size_t j = 0; j < n; ++j) a[j] = b[j] + q * c[j];
        t = now_seconds() - t;
        if (k > 0 && t < best_triad) best_triad = t;
    }

    out->validated = stream_validate(a, b, c, n, ntimes);
    if (!out->validated) g_validation_failed = 1;

    /* MB/s on STREAM's 1e6 convention. */
    double two = 2.0 * sizeof(double) * (double)n / 1.0e6;
    double three = 3.0 * sizeof(double) * (double)n / 1.0e6;
    out->copy  = best_copy  < 1e29 ? two   / best_copy  : 0.0;
    out->scale = best_scale < 1e29 ? two   / best_scale : 0.0;
    out->add   = best_add   < 1e29 ? three / best_add   : 0.0;
    out->triad = best_triad < 1e29 ? three / best_triad : 0.0;
    out->elements = n;
    out->array_bytes = n * sizeof(double);

    g_sink_d += a[0] + b[n / 2] + c[n - 1];
    free(a); free(b); free(c);
    return 1;
}

/* --------------------------- CoreMark-style ----------------------------
 * EEMBC's CoreMark is the reference integer benchmark for embedded and
 * general-purpose cores, and it exists because Dhrystone was trivially
 * defeated by compilers. It combines four kernels whose results feed each
 * other, so none can be optimised away in isolation:
 *
 *   - linked-list insert / find / sort
 *   - small matrix multiply and column operations
 *   - a finite state machine over a character buffer
 *   - CRC over every intermediate result, which is also the validation
 *
 * This is a faithful *reimplementation of the same kernel mix*, not the
 * certified benchmark. Published CoreMark scores come from EEMBC's exact
 * source under strict reporting rules (fixed run duration, disclosed compiler
 * flags, no library calls in the timed region), and a number produced here
 * must not be presented as a CoreMark score. It is reported as
 * "coremark_style" throughout for that reason. What it is good for is
 * comparing cores to each other under an identical, compiler-resistant
 * integer workload — which is most of what people want it for.
 */
#define CM_LIST_SIZE   200
#define CM_MATRIX_DIM  16
#define CM_FSM_LEN     512

typedef struct cm_node {
    struct cm_node *next;
    short value;
    short info;
} cm_node;

static unsigned short cm_crc16(unsigned short crc, unsigned short data) {
    for (int i = 0; i < 16; ++i) {
        unsigned short bit = ((data >> i) & 1) ^ (crc & 1);
        crc >>= 1;
        if (bit) crc ^= 0xA001;
    }
    return crc;
}

/* Insertion sort on the linked list, by value then info. */
static cm_node *cm_sort(cm_node *head) {
    cm_node *sorted = NULL;
    while (head) {
        cm_node *next = head->next;
        if (!sorted || head->value < sorted->value) {
            head->next = sorted;
            sorted = head;
        } else {
            cm_node *p = sorted;
            while (p->next && p->next->value <= head->value) p = p->next;
            head->next = p->next;
            p->next = head;
        }
        head = next;
    }
    return sorted;
}

static unsigned short cm_list_kernel(cm_node *pool, unsigned short seed) {
    for (int i = 0; i < CM_LIST_SIZE; ++i) {
        pool[i].value = (short)((seed + i * 7) & 0x7FFF);
        pool[i].info = (short)(i & 0xFF);
        pool[i].next = (i + 1 < CM_LIST_SIZE) ? &pool[i + 1] : NULL;
    }
    cm_node *head = cm_sort(&pool[0]);
    unsigned short crc = 0;
    int found = 0;
    for (cm_node *p = head; p; p = p->next) {
        crc = cm_crc16(crc, (unsigned short)p->value);
        if (p->next && p->next->value < p->value) found++;   /* must stay 0 */
    }
    if (found != 0) g_validation_failed = 1;
    return crc;
}

static unsigned short cm_matrix_kernel(short *a, short *b, int *out,
                                       unsigned short seed) {
    const int N = CM_MATRIX_DIM;
    for (int i = 0; i < N * N; ++i) {
        a[i] = (short)((seed + i) & 0xFF);
        b[i] = (short)((seed ^ i) & 0xFF);
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += (int)a[i * N + k] * (int)b[k * N + j];
            out[i * N + j] = sum;
        }
    unsigned short crc = 0;
    for (int i = 0; i < N * N; i += 7)
        crc = cm_crc16(crc, (unsigned short)(out[i] & 0xFFFF));
    return crc;
}

/* Four-state machine classifying a character buffer, as CoreMark's does. */
static unsigned short cm_fsm_kernel(unsigned char *buf, unsigned short seed) {
    for (int i = 0; i < CM_FSM_LEN; ++i)
        buf[i] = (unsigned char)((seed + i * 31) & 0x7F);

    int state = 0;
    unsigned int counts[4] = {0, 0, 0, 0};
    for (int i = 0; i < CM_FSM_LEN; ++i) {
        unsigned char ch = buf[i];
        switch (state) {
            case 0: state = (ch >= '0' && ch <= '9') ? 1 : 2; break;
            case 1: state = (ch == '.') ? 3 : ((ch >= '0' && ch <= '9') ? 1 : 0); break;
            case 2: state = (ch == ' ') ? 0 : 2; break;
            default: state = (ch >= '0' && ch <= '9') ? 3 : 0; break;
        }
        counts[state]++;
    }
    unsigned short crc = 0;
    for (int i = 0; i < 4; ++i)
        crc = cm_crc16(crc, (unsigned short)(counts[i] & 0xFFFF));
    return crc;
}

static double run_coremark_style(double seconds, unsigned short *out_crc) {
    cm_node *pool = (cm_node *)malloc(sizeof(cm_node) * CM_LIST_SIZE);
    short *ma = (short *)malloc(sizeof(short) * CM_MATRIX_DIM * CM_MATRIX_DIM);
    short *mb = (short *)malloc(sizeof(short) * CM_MATRIX_DIM * CM_MATRIX_DIM);
    int *mo = (int *)malloc(sizeof(int) * CM_MATRIX_DIM * CM_MATRIX_DIM);
    unsigned char *fsm = (unsigned char *)malloc(CM_FSM_LEN);
    if (!pool || !ma || !mb || !mo || !fsm) {
        free(pool); free(ma); free(mb); free(mo); free(fsm);
        return 0.0;
    }

    unsigned short crc = 0, seed = 0x1234;
    long long iterations = 0;
    double start = now_seconds(), elapsed;
    do {
        /* Each kernel's CRC seeds the next, so the chain cannot be
         * reordered or elided. */
        crc = cm_crc16(crc, cm_list_kernel(pool, seed));
        crc = cm_crc16(crc, cm_matrix_kernel(ma, mb, mo, crc));
        crc = cm_crc16(crc, cm_fsm_kernel(fsm, crc));
        seed = crc;
        ++iterations;
        elapsed = now_seconds() - start;
    } while (elapsed < seconds);

    g_sink_i += crc;
    *out_crc = crc;
    free(pool); free(ma); free(mb); free(mo); free(fsm);
    return (double)iterations / elapsed;
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
    /* Two buffers are allocated, so cap each at 1/16 of RAM to keep the
     * combined footprint under 1/8 — far clear of swap. */
    unsigned long long ram = total_ram_bytes();
    if (ram) {
        size_t cap = (size_t)(ram / (RAM_SAFE_FRACTION * 2));
        if (n > cap) n = cap;
    }
    if (n < (size_t)MB) n = MB;
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
    /* Skip working sets this machine cannot hold comfortably: a 256 MB chase
     * on a 512 MB board would thrash swap and measure the disk, not the cache. */
    unsigned long long ram = total_ram_bytes();
    if (ram && bytes > (size_t)(ram / RAM_SAFE_FRACTION)) return 0.0;

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
/* Where the disk test writes. Empty means "the system temp directory", which
 * is what this used to do unconditionally -- and on most Linux systems /tmp is
 * tmpfs, so the whole test wrote to RAM and reported memory bandwidth as
 * storage throughput. The Python side passes the same directory the rest of
 * the tool writes to, which is on real storage. */
static const char *g_disk_dir = "";

static void run_disk(int file_mb, double *out_write, double *out_read) {
    size_t chunk = 4 * (size_t)MB;
    long n_chunks = (long)(((size_t)file_mb * MB) / chunk);
    if (n_chunks < 1) n_chunks = 1;
    size_t total = (size_t)n_chunks * chunk;

    /* Refuse to write if the target filesystem lacks headroom. This matters
     * most on Linux, where /tmp is frequently tmpfs — writing there consumes
     * RAM, not storage, and filling it can destabilize the whole system. */
#if defined(_WIN32)
    char probe[MAX_PATH];
    if (g_disk_dir[0]) snprintf(probe, sizeof(probe), "%s", g_disk_dir);
    else GetTempPathA(MAX_PATH, probe);
#else
    const char *probe = g_disk_dir[0] ? g_disk_dir : "/tmp";
#endif
    unsigned long long freeb = free_disk_bytes(probe);
    if (freeb && (double)total * DISK_HEADROOM > (double)freeb) {
        *out_write = *out_read = 0.0;
        return;                      /* reported as 0; never fills the disk */
    }

    char *buf = (char *)malloc(chunk);
    if (!buf) { *out_write = *out_read = 0.0; return; }
    /* Incompressible. A buffer of one repeated byte costs nothing to store on
     * anything that compresses -- btrfs with compress=, ZFS, NTFS compression,
     * and the inline compression in many SSD controllers -- so the write never
     * reaches the medium at the rate being reported. A cheap xorshift fill is
     * enough: it need not be cryptographic, only incompressible. */
    {
        unsigned long long x = 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < chunk; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            buf[i] = (char)(x & 0xFF);
        }
    }

    char path[512];
#if defined(_WIN32)
    char tmpdir[MAX_PATH];
    if (g_disk_dir[0]) snprintf(tmpdir, sizeof(tmpdir), "%s\\", g_disk_dir);
    else GetTempPathA(MAX_PATH, tmpdir);
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
    snprintf(path, sizeof(path), "%s/native_bench_XXXXXX",
             g_disk_dir[0] ? g_disk_dir : "/tmp");
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

static void print_human_standards(const StreamResult *st, double coremark) {
    if (st->elements) {
        printf("\n  STREAM (%.0f MB per array, MB/s on the 1e6 convention):\n",
               (double)st->array_bytes / 1.0e6);
        printf("    Copy  : %10.1f\n    Scale : %10.1f\n"
               "    Add   : %10.1f\n    Triad : %10.1f%s\n",
               st->copy, st->scale, st->add, st->triad,
               st->validated ? "" : "   (VALIDATION FAILED)");
    }
    if (coremark > 0.0)
        printf("\n  CoreMark-style : %10.1f iterations/s "
               "(not a certified CoreMark score)\n", coremark);
}

static void print_json(Res *r, int n, double *lat,
                       double seconds, int repeats, int threads,
                       const StreamResult *st, double coremark,
                       unsigned crc) {
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
    printf("  ],\n");

    printf("  \"stream\": ");
    if (st->elements) {
        printf("{\"unit\": \"MB/s\", \"convention\": \"1e6 bytes\", "
               "\"array_bytes\": %zu, \"elements\": %zu, "
               "\"validated\": %s, \"copy\": %.2f, \"scale\": %.2f, "
               "\"add\": %.2f, \"triad\": %.2f},\n",
               st->array_bytes, st->elements, st->validated ? "true" : "false",
               st->copy, st->scale, st->add, st->triad);
    } else {
        printf("null,\n");
    }

    printf("  \"coremark_style\": ");
    if (coremark > 0.0)
        printf("{\"unit\": \"iterations/s\", \"rate\": %.3f, "
               "\"crc\": %u, \"certified\": false, "
               "\"note\": \"same kernel mix as EEMBC CoreMark; not the "
               "certified benchmark and not comparable to published "
               "CoreMark scores\"}\n", coremark, crc);
    else
        printf("null\n");

    printf("}\n");
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--json] [--seconds N] [--repeats M] [--threads T]\n"
            "          [--mem-mb K] [--disk-mb K] [--disk-dir PATH]\n"
            "          [--stream-mb K]\n"
            "          [--no-standards]\n", prog);
}

/* --------------------------- Main -------------------------------------- */
int main(int argc, char **argv) {
    double seconds = 3.0;
    int repeats = 3, as_json = 0, threads = 0, mem_mb = 64, disk_mb = 256;
    int stream_mb = 0, no_standards = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json")) as_json = 1;
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-mb")  && i + 1 < argc) mem_mb  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--disk-mb") && i + 1 < argc) disk_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--disk-dir") && i + 1 < argc) g_disk_dir = argv[++i];
        else if (!strcmp(argv[i], "--stream-mb") && i + 1 < argc) stream_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-standards")) no_standards = 1;
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

    /* STREAM and the CoreMark-style suite: reference workloads whose value
     * is that their numbers mean something outside this tool. */
    StreamResult stream;
    memset(&stream, 0, sizeof(stream));
    double coremark = 0.0;
    unsigned short cm_crc = 0;

    if (!no_standards) {
        /* Each array must clear the last-level cache by a wide margin, and
         * three of them are allocated at once. 64 MB apiece satisfies the 4x
         * rule for any cache up to 16 MB while staying under 1/5 of RAM on a
         * 1 GB board. */
        size_t bytes;
        if (stream_mb > 0) {
            bytes = (size_t)stream_mb * MB;
        } else {
            unsigned long long ram = total_ram_bytes();
            bytes = 64 * (size_t)MB;
            if (ram > 0 && bytes * 3 > ram / 4) bytes = (size_t)(ram / 12);
            if (bytes < 4 * (size_t)MB) bytes = 4 * (size_t)MB;
        }
        int ntimes = (int)(seconds * 2.0);
        if (ntimes < 3) ntimes = 3;
        if (ntimes > 20) ntimes = 20;
        run_stream(&stream, bytes, ntimes);
        coremark = run_coremark_style(seconds, &cm_crc);
    }

    Res results[6];
    results[0] = (Res){"CPU Integer (primes)",  "primes/s", median(ci, repeats), stddev(ci, repeats)};
    results[1] = (Res){"CPU Float (math ops)",  "iters/s",  median(cf, repeats), stddev(cf, repeats)};
    results[2] = (Res){"CPU Multi-thread",      "primes/s", median(mt, repeats), stddev(mt, repeats)};
    results[3] = (Res){"Memory copy bandwidth", "MB/s",     median(mm, repeats), stddev(mm, repeats)};
    results[4] = (Res){"Disk write",            "MB/s",     median(dw, repeats), stddev(dw, repeats)};
    results[5] = (Res){"Disk read",             "MB/s",     median(dr, repeats), stddev(dr, repeats)};

    if (as_json) {
        print_json(results, 6, lat, seconds, repeats, threads,
                   &stream, coremark, cm_crc);
    } else {
        print_human(results, 6, lat);
        print_human_standards(&stream, coremark);
    }

    free(ci); free(cf); free(mt); free(mm); free(dw); free(dr);
    if (g_sink_i == 0x7FFFFFFF && g_sink_d == 1.5) fputs("", stderr);
    return g_validation_failed ? 2 : 0;
}
