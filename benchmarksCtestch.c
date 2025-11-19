// benchmarksCtestch.c
// Simple cross-platform benchmark: CPU int, CPU float, memory, disk I/O.
//
// Build (macOS, Linux):
//   gcc -O2 benchmarksCtestch.c -o benchmarksCtestch -lm
// or
//   clang -O2 benchmarksCtestch.c -o benchmarksCtestch -lm
//
// Run:
//   ./benchmarksCtestch
//   ./benchmarksCtestch --seconds 5 --repeats 5

#define _POSIX_C_SOURCE 199309L
#define _DARWIN_C_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

// --------------------------- Timing helpers ---------------------------

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        // Fallback: should not happen on modern macOS/Linux
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// --------------------------- Math helpers -----------------------------

static int is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    int r = (int) sqrt((double)n);
    for (int i = 3; i <= r; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

// --------------------------- Workload chunks --------------------------

// Integer-heavy workload: primality checks
static void cpu_integer_chunk(void) {
    for (int n = 50000; n < 51000; ++n) {
        (void) is_prime(n);
    }
}

// Floating-point workload: sin/cos/sqrt in tight loop
static void cpu_float_chunk(void) {
    double x = 0.001;
    double s = 0.0;
    for (int i = 0; i < 50000; ++i) {
        x += 0.00001;
        s += sin(x) * cos(x) + sqrt(x);
    }
    // Prevent compiler from optimizing everything away
    if (s == -1.23456789) {
        fprintf(stderr, "magic float %f\n", s);
    }
}

// Memory workload: allocate, fill, sum
static void memory_chunk(void) {
    const size_t size = 500000; // ~ 4 MB in doubles
    double *data = (double *) malloc(size * sizeof(double));
    if (!data) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] = (double)i;
    }
    double total = 0.0;
    for (size_t i = 0; i < size; ++i) {
        total += data[i];
    }
    if (total == -1.23456789) {
        fprintf(stderr, "magic mem %f\n", total);
    }
    free(data);
}

// Disk workload: write & read a temporary ~5 MB file
static void disk_chunk(void) {
    const size_t size = 5 * 1024 * 1024; // 5 MB
    char *buf = (char *) malloc(size);
    if (!buf) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(buf, 'X', size);

    char tmpl[] = "c_bench_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd == -1) {
        perror("mkstemp");
        free(buf);
        exit(EXIT_FAILURE);
    }

    // Write
    ssize_t written = write(fd, buf, size);
    if (written < 0) {
        perror("write");
    }

    if (fsync(fd) != 0) {
        perror("fsync");
    }

    // Read
    if (lseek(fd, 0, SEEK_SET) == (off_t)-1) {
        perror("lseek");
    }
    ssize_t read_bytes = read(fd, buf, size);
    if (read_bytes < 0) {
        perror("read");
    }

    close(fd);
    unlink(tmpl);
    free(buf);
}

// --------------------------- Benchmark core ---------------------------

typedef void (*chunk_func_t)(void);

typedef struct {
    const char *name;
    double avg_cps;
    double stddev_cps;
} Result;

static void print_header(const char *title) {
    printf("\n======================================================================\n");
    printf("%s\n", title);
    printf("======================================================================\n");
}

static void run_benchmark_chunked(chunk_func_t func,
                                  double seconds,
                                  int repeats,
                                  Result *out_result) {
    double *cps_vals = (double *) malloc(repeats * sizeof(double));
    if (!cps_vals) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }

    print_header(out_result->name);

    for (int r = 0; r < repeats; ++r) {
        double start = now_seconds();
        double elapsed = 0.0;
        long long count = 0;

        while (1) {
            func();
            ++count;
            elapsed = now_seconds() - start;
            if (elapsed >= seconds) break;
        }

        double cps = (elapsed > 0.0) ? (double)count / elapsed : 0.0;
        cps_vals[r] = cps;
        printf("  Run %d: %.2f s, %lld chunks -> %.2f chunks/s\n",
               r + 1, elapsed, count, cps);
    }

    // compute mean
    double sum = 0.0;
    for (int i = 0; i < repeats; ++i) {
        sum += cps_vals[i];
    }
    double mean = sum / repeats;

    // compute std dev
    double var_sum = 0.0;
    for (int i = 0; i < repeats; ++i) {
        double diff = cps_vals[i] - mean;
        var_sum += diff * diff;
    }
    double stddev = 0.0;
    if (repeats > 1) {
        stddev = sqrt(var_sum / (repeats - 1));
    }

    out_result->avg_cps = mean;
    out_result->stddev_cps = stddev;

    free(cps_vals);
}

// --------------------------- System info ------------------------------

static void print_system_info(void) {
    struct utsname u;
    if (uname(&u) != 0) {
        perror("uname");
        return;
    }

    print_header("System Information");
    printf("sysname   : %s\n", u.sysname);   // "Darwin" or "Linux"
    printf("nodename  : %s\n", u.nodename);
    printf("release   : %s\n", u.release);
    printf("version   : %s\n", u.version);
    printf("machine   : %s\n", u.machine);   // "arm64", "x86_64", etc.
}

// --------------------------- Main ------------------------------------

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [--seconds N] [--repeats M]\n"
            "  --seconds N  Target duration per test per repeat (default 3.0)\n"
            "  --repeats M  Number of repeats per test (default 3)\n",
            prog);
}

int main(int argc, char **argv) {
    double seconds_per_test = 3.0;
    int repeats = 3;

    // Simple arg parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--seconds") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            seconds_per_test = atof(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0) {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            repeats = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    print_system_info();

    Result results[4] = {
        { "CPU Integer (primes)", 0.0, 0.0 },
        { "CPU Float (math ops)", 0.0, 0.0 },
        { "Memory (alloc & sum)", 0.0, 0.0 },
        { "Disk I/O (5 MB R/W)",  0.0, 0.0 }
    };

    run_benchmark_chunked(cpu_integer_chunk, seconds_per_test, repeats, &results[0]);
    run_benchmark_chunked(cpu_float_chunk,   seconds_per_test, repeats, &results[1]);
    run_benchmark_chunked(memory_chunk,      seconds_per_test, repeats, &results[2]);
    run_benchmark_chunked(disk_chunk,        seconds_per_test, repeats, &results[3]);

    print_header("Summary (higher is better)");
    for (int i = 0; i < 4; ++i) {
        printf("%-24s: %10.2f chunks/s  (std dev: %.2f)\n",
               results[i].name, results[i].avg_cps, results[i].stddev_cps);
    }

    // JSON-ish snapshot line (easy to copy into notes)
    struct utsname u;
    uname(&u);

    print_header("Machine Score Snapshot");
    printf("{\n");
    printf("  \"system\": \"%s\",\n", u.sysname);
    printf("  \"machine\": \"%s\",\n", u.machine);
    printf("  \"results\": {\n");
    for (int i = 0; i < 4; ++i) {
        printf("    \"%s\": %.4f%s\n",
               results[i].name,
               results[i].avg_cps,
               (i == 3) ? "" : ",");
    }
    printf("  }\n");
    printf("}\n");

    return 0;
}
