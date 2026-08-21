/*
 * accel_engine.m — GPU and Neural Engine benchmarks for Apple platforms.
 *
 * Two accelerators, two very different APIs:
 *
 *   GPU  — Metal compute kernels. The shader source is compiled at *runtime*
 *          via newLibraryWithSource:, deliberately, so the engine builds with
 *          only the Command Line Tools; the offline `metal` compiler ships
 *          with full Xcode, which most machines do not have.
 *
 *   ANE  — Core ML. The Neural Engine cannot be targeted directly; there is no
 *          public API to submit arbitrary work to it. Core ML alone decides
 *          placement, so the benchmark runs the same model under different
 *          MLComputeUnits settings and reports the speedup, which is the only
 *          honest evidence that the ANE was actually used.
 *
 * Build:
 *   clang -O2 -fobjc-arc accel_engine.m -o accel_engine \
 *         -framework Foundation -framework Metal -framework CoreML
 *
 * Run:
 *   ./accel_engine --json --seconds 2 --model /path/to/ane_model.mlmodel
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreML/CoreML.h>
#import <mach/mach_time.h>

/* --------------------------- Timing ------------------------------------ */
static double now_seconds(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

/* --------------------------- Metal shaders ----------------------------- */
/* Each thread runs a long dependent FMA chain. The chain is dependent so the
 * compiler cannot vectorize it away, and the result is written out so the
 * whole kernel cannot be eliminated as dead code. */
static NSString *const kShaderSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void fma_f32(device float *out [[buffer(0)]],\n"
"                    constant uint &iters [[buffer(1)]],\n"
"                    uint gid [[thread_position_in_grid]]) {\n"
"    float a = out[gid], b = 1.0000001f, c = 0.0000001f;\n"
"    float a2 = a + 1.0f, a3 = a + 2.0f, a4 = a + 3.0f;\n"
"    for (uint k = 0; k < iters; ++k) {\n"
"        a  = fma(a,  b, c); a2 = fma(a2, b, c);\n"
"        a3 = fma(a3, b, c); a4 = fma(a4, b, c);\n"
"    }\n"
"    out[gid] = a + a2 + a3 + a4;\n"
"}\n"
"\n"
"kernel void fma_f16(device half *out [[buffer(0)]],\n"
"                    constant uint &iters [[buffer(1)]],\n"
"                    uint gid [[thread_position_in_grid]]) {\n"
"    half a = out[gid], b = 1.0h, c = 0.0001h;\n"
"    half a2 = a + 1.0h, a3 = a + 2.0h, a4 = a + 3.0h;\n"
"    for (uint k = 0; k < iters; ++k) {\n"
"        a  = fma(a,  b, c); a2 = fma(a2, b, c);\n"
"        a3 = fma(a3, b, c); a4 = fma(a4, b, c);\n"
"    }\n"
"    out[gid] = a + a2 + a3 + a4;\n"
"}\n"
"\n"
"kernel void bandwidth(device const float4 *src [[buffer(0)]],\n"
"                      device float4 *dst [[buffer(1)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    dst[gid] = src[gid];\n"
"}\n"
"\n"
"kernel void nop(device float *out [[buffer(0)]],\n"
"                uint gid [[thread_position_in_grid]]) {\n"
"    if (gid == 0) out[0] += 1.0f;\n"
"}\n";

/* --------------------------- JSON helpers ------------------------------ */
static NSMutableArray *g_results;   /* array of @{name, unit, value} */
static NSMutableArray *g_notes;

static void add_result(NSString *name, NSString *unit, double value) {
    [g_results addObject:@{@"name": name, @"unit": unit, @"value": @(value)}];
}
static void add_note(NSString *note) { [g_notes addObject:note]; }

/* --------------------------- Matrix multiply (AI compute) -------------- */
/* Dense GEMM is the operation that dominates neural-network compute — every
 * fully-connected and convolution layer reduces to it — so its sustained
 * TFLOPS is the single most meaningful "AI performance" number for a GPU.
 * MetalPerformanceShaders provides a hand-tuned kernel, so this measures the
 * hardware rather than our shader-writing. */
static double run_matmul(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                         int N, MPSDataType dtype, double seconds) {
    @autoreleasepool {
        size_t elem = (dtype == MPSDataTypeFloat16) ? 2 : 4;
        MPSMatrixDescriptor *desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:N
                                                 rowBytes:N * elem
                                                 dataType:dtype];
        /* Three NxN matrices must fit comfortably in the GPU's working set. */
        NSUInteger need = (NSUInteger)N * N * elem * 3;
        NSUInteger budget = (NSUInteger)dev.recommendedMaxWorkingSetSize;
        if (budget > 0 && need > budget / 4) return 0.0;

        id<MTLBuffer> ba = [dev newBufferWithLength:(NSUInteger)N * N * elem
                                  options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bb = [dev newBufferWithLength:(NSUInteger)N * N * elem
                                  options:MTLResourceStorageModePrivate];
        id<MTLBuffer> bc = [dev newBufferWithLength:(NSUInteger)N * N * elem
                                  options:MTLResourceStorageModePrivate];
        if (!ba || !bb || !bc) return 0.0;
        MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:ba descriptor:desc];
        MPSMatrix *B = [[MPSMatrix alloc] initWithBuffer:bb descriptor:desc];
        MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:bc descriptor:desc];
        MPSMatrixMultiplication *mm =
            [[MPSMatrixMultiplication alloc] initWithDevice:dev
                transposeLeft:NO transposeRight:NO resultRows:N
                resultColumns:N interiorColumns:N alpha:1.0 beta:0.0];

        for (int i = 0; i < 3; ++i) {          /* warm up */
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B
                         resultMatrix:C];
            [cb commit];
            [cb waitUntilCompleted];
        }

        double start = now_seconds(), elapsed = 0;
        long long count = 0;
        do {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            /* Batch several multiplies per command buffer so submission
             * overhead does not dominate at small N. */
            for (int b = 0; b < 8; ++b)
                [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B
                             resultMatrix:C];
            [cb commit];
            [cb waitUntilCompleted];
            count += 8;
            elapsed = now_seconds() - start;
        } while (elapsed < seconds);

        /* A dense NxN multiply is 2*N^3 FLOPs (one multiply + one add each). */
        double flops = (double)count * 2.0 * (double)N * N * N;
        return flops / elapsed / 1e12;      /* TFLOPS */
    }
}

/* --------------------------- GPU benchmarks ---------------------------- */
static NSDictionary *run_gpu(double seconds) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { add_note(@"no Metal device available"); return nil; }

    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kShaderSource
                                           options:nil error:&err];
    if (!lib) {
        add_note([NSString stringWithFormat:@"Metal shader compile failed: %@",
                  err.localizedDescription]);
        return nil;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSMutableDictionary *info = [@{
        @"name": dev.name ?: @"unknown",
        @"unified_memory": @(dev.hasUnifiedMemory),
        @"max_threads_per_group": @(dev.maxThreadsPerThreadgroup.width),
        @"recommended_working_set_mb":
            @(dev.recommendedMaxWorkingSetSize / (1024.0 * 1024.0)),
    } mutableCopy];
    if (@available(macOS 10.15, *))
        info[@"registry_id"] = @(dev.registryID);

    const NSUInteger threads = 1 << 20;          /* 1M threads */
    const uint32_t inner = 2048;                 /* FMA loop trips */

    /* ---- FP32 / FP16 FMA throughput ---- */
    NSArray *kernels = @[@"fma_f32", @"fma_f16"];
    NSArray *labels  = @[@"GPU FP32 FMA", @"GPU FP16 FMA"];
    NSArray *elemSz  = @[@4, @2];
    for (NSUInteger k = 0; k < kernels.count; ++k) {
        id<MTLFunction> fn = [lib newFunctionWithName:kernels[k]];
        id<MTLComputePipelineState> pipe =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) continue;

        id<MTLBuffer> buf =
            [dev newBufferWithLength:threads * [elemSz[k] unsignedIntValue]
                             options:MTLResourceStorageModePrivate];
        uint32_t iters = inner;

        NSUInteger tg = MIN(pipe.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        double start = now_seconds(), elapsed = 0;
        long long dispatches = 0;
        do {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setBytes:&iters length:sizeof(iters) atIndex:1];
            [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            ++dispatches;
            elapsed = now_seconds() - start;
        } while (elapsed < seconds);

        /* 4 independent chains x 2 FLOPs per fma x inner trips x threads */
        double flops = (double)dispatches * threads * inner * 4.0 * 2.0;
        add_result(labels[k], @"GFLOPS", flops / elapsed / 1e9);
    }

    /* ---- Matrix-multiply throughput (the core AI-compute metric) ---- */
    {
        const int N = 2048;                    /* large enough to saturate */
        double f32 = run_matmul(dev, queue, N, MPSDataTypeFloat32,
                                MAX(seconds * 0.5, 0.5));
        double f16 = run_matmul(dev, queue, N, MPSDataTypeFloat16,
                                MAX(seconds * 0.5, 0.5));
        if (f32 > 0) add_result(@"GPU matmul FP32 (GEMM)", @"TFLOPS", f32);
        if (f16 > 0) add_result(@"GPU matmul FP16 (GEMM)", @"TFLOPS", f16);
    }

    /* ---- Device memory bandwidth ---- */
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"bandwidth"];
        id<MTLComputePipelineState> pipe =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (pipe) {
            /* Clamp against what the device says it can hold. On Apple
             * silicon the GPU shares system RAM, so an oversized pair of
             * buffers would take memory from the OS and applications. */
            NSUInteger bytes = 256u * 1024u * 1024u;          /* 256 MB */
            NSUInteger budget = (NSUInteger)dev.recommendedMaxWorkingSetSize;
            if (budget > 0 && bytes * 2 > budget / 4)
                bytes = (budget / 4) / 2;
            if (bytes < 16u * 1024u * 1024u) bytes = 16u * 1024u * 1024u;
            bytes &= ~(NSUInteger)15;                  /* float4 alignment */
            const NSUInteger vecs = bytes / 16;               /* float4 */
            id<MTLBuffer> src = [dev newBufferWithLength:bytes
                                    options:MTLResourceStorageModePrivate];
            id<MTLBuffer> dst = [dev newBufferWithLength:bytes
                                    options:MTLResourceStorageModePrivate];
            NSUInteger tg = MIN(pipe.maxTotalThreadsPerThreadgroup,
                                (NSUInteger)256);
            double start = now_seconds(), elapsed = 0;
            long long passes = 0;
            do {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe];
                [enc setBuffer:src offset:0 atIndex:0];
                [enc setBuffer:dst offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(vecs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                ++passes;
                elapsed = now_seconds() - start;
            } while (elapsed < seconds);
            /* One read plus one write per element. */
            double moved = (double)passes * bytes * 2.0;
            add_result(@"GPU memory bandwidth", @"MB/s",
                       moved / elapsed / (1024.0 * 1024.0));
        }
    }

    /* ---- Kernel launch latency ---- */
    {
        id<MTLFunction> fn = [lib newFunctionWithName:@"nop"];
        id<MTLComputePipelineState> pipe =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (pipe) {
            id<MTLBuffer> buf = [dev newBufferWithLength:1024
                                    options:MTLResourceStorageModePrivate];
            double budget = MIN(seconds, 1.0);
            double start = now_seconds(), elapsed = 0;
            long long n = 0;
            do {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                ++n;
                elapsed = now_seconds() - start;
            } while (elapsed < budget);
            add_result(@"GPU kernel launch latency", @"us",
                       elapsed / (double)n * 1e6);
        }
    }
    return info;
} }

/* --------------------------- Neural Engine ----------------------------- */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Percentile (0..100) of an unsorted array of per-inference milliseconds.
 * Tail latency (p99) matters for interactive AI: a good average with a bad
 * p99 still produces visible stutter. */
static double percentile(double *v, long n, double p) {
    if (n <= 0) return 0.0;
    qsort(v, n, sizeof(double), cmp_double);
    long idx = (long)(p / 100.0 * (n - 1) + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return v[idx];
}

/* Runs the model under one MLComputeUnits setting; returns inferences/sec.
 * When `lat_ms` is non-NULL it also records each inference's latency (ms) so
 * the caller can compute p50/p99. */
static double run_coreml(NSURL *compiled, MLComputeUnits units,
                         NSArray<NSNumber *> *shape, double seconds,
                         BOOL *ok, NSMutableArray<NSNumber *> *lat_ms) {
    @autoreleasepool {
        NSError *err = nil;
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = units;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:cfg error:&err];
        if (!model) { *ok = NO; return 0.0; }

        MLMultiArray *input =
            [[MLMultiArray alloc] initWithShape:shape
                                       dataType:MLMultiArrayDataTypeFloat32
                                          error:&err];
        if (!input) { *ok = NO; return 0.0; }
        float *p = (float *)input.dataPointer;
        NSUInteger count = input.count;
        for (NSUInteger i = 0; i < count; ++i) p[i] = 0.5f;

        MLDictionaryFeatureProvider *feed =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{@"input": input} error:&err];
        if (!feed) { *ok = NO; return 0.0; }

        /* Warm-up: the first inferences pay model load, weight conversion,
         * and ANE program compilation, none of which belong in the result. */
        for (int i = 0; i < 5; ++i)
            [model predictionFromFeatures:feed error:&err];

        double start = now_seconds(), elapsed = 0;
        long long n = 0;
        do {
            @autoreleasepool {
                double t0 = now_seconds();
                if (![model predictionFromFeatures:feed error:&err]) {
                    *ok = NO;
                    return 0.0;
                }
                if (lat_ms)
                    [lat_ms addObject:@((now_seconds() - t0) * 1000.0)];
            }
            ++n;
            elapsed = now_seconds() - start;
        } while (elapsed < seconds);

        *ok = YES;
        return (double)n / elapsed;
    }
}

static NSDictionary *run_ane(NSString *modelPath, double seconds,
                             double flopsPerInference,
                             NSArray<NSNumber *> *shape) {
    @autoreleasepool {
        if (!modelPath || ![[NSFileManager defaultManager]
                             fileExistsAtPath:modelPath]) {
            add_note(@"no Core ML model supplied; ANE benchmark skipped");
            return nil;
        }
        NSError *err = nil;
        NSURL *compiled =
            [MLModel compileModelAtURL:[NSURL fileURLWithPath:modelPath]
                                 error:&err];
        if (!compiled) {
            add_note([NSString stringWithFormat:
                      @"Core ML model compile failed: %@",
                      err.localizedDescription]);
            return nil;
        }

        BOOL ok = NO;
        double cpu = run_coreml(compiled, MLComputeUnitsCPUOnly, shape,
                                seconds, &ok, nil);
        if (!ok) { add_note(@"Core ML CPU-only run failed"); return nil; }

        NSMutableArray<NSNumber *> *aneLat = [NSMutableArray array];
        double ane = run_coreml(compiled, MLComputeUnitsCPUAndNeuralEngine,
                                shape, seconds, &ok, aneLat);
        if (!ok) { add_note(@"Core ML ANE run failed"); ane = 0.0; }

        double all = run_coreml(compiled, MLComputeUnitsAll, shape,
                                seconds, &ok, nil);
        if (!ok) all = 0.0;

        add_result(@"Core ML CPU-only", @"inferences/s", cpu);
        add_result(@"Neural Engine", @"inferences/s", ane);
        add_result(@"Core ML best (CPU+GPU+ANE)", @"inferences/s", all);
        if (ane > 0)
            add_result(@"Neural Engine throughput", @"GFLOPS",
                       ane * flopsPerInference / 1e9);

        /* Tail latency on the ANE path. */
        double p50 = 0, p99 = 0;
        long nl = (long)aneLat.count;
        if (nl > 0) {
            double *tmp = (double *)malloc(sizeof(double) * nl);
            for (long i = 0; i < nl; ++i) tmp[i] = aneLat[i].doubleValue;
            p50 = percentile(tmp, nl, 50.0);
            p99 = percentile(tmp, nl, 99.0);
            free(tmp);
            add_result(@"Neural Engine latency p50", @"ms", p50);
            add_result(@"Neural Engine latency p99", @"ms", p99);
        }

        double speedup = (cpu > 0) ? ane / cpu : 0.0;
        /* Core ML never reports placement directly, so the speedup over a
         * CPU-only run is the evidence. Below ~1.5x the model was almost
         * certainly kept on the CPU and the number means nothing. */
        BOOL engaged = speedup >= 1.5;
        if (!engaged)
            add_note(@"Neural Engine did not engage (speedup < 1.5x); "
                     @"Core ML likely kept this model on the CPU");

        return @{@"cpu_only_ips": @(cpu),
                 @"ane_ips": @(ane),
                 @"best_ips": @(all),
                 @"speedup_vs_cpu": @(speedup),
                 @"latency_p50_ms": @(p50),
                 @"latency_p99_ms": @(p99),
                 @"engaged": @(engaged)};
    }
}

/* --------------------------- Main -------------------------------------- */
int main(int argc, const char *argv[]) { @autoreleasepool {
    double seconds = 2.0;
    BOOL asJson = NO, doGpu = YES, doAne = YES;
    NSString *modelPath = nil;
    double flops = 0.0;
    NSMutableArray<NSNumber *> *shape = [@[@1, @64, @64, @64] mutableCopy];

    for (int i = 1; i < argc; ++i) {
        NSString *a = @(argv[i]);
        if ([a isEqualToString:@"--json"]) asJson = YES;
        else if ([a isEqualToString:@"--no-gpu"]) doGpu = NO;
        else if ([a isEqualToString:@"--no-ane"]) doAne = NO;
        else if ([a isEqualToString:@"--seconds"] && i + 1 < argc)
            seconds = atof(argv[++i]);
        else if ([a isEqualToString:@"--model"] && i + 1 < argc)
            modelPath = @(argv[++i]);
        else if ([a isEqualToString:@"--flops"] && i + 1 < argc)
            flops = atof(argv[++i]);
        else if ([a isEqualToString:@"--shape"] && i + 1 < argc) {
            [shape removeAllObjects];
            for (NSString *part in [@(argv[++i]) componentsSeparatedByString:@","])
                [shape addObject:@([part integerValue])];
        } else {
            fprintf(stderr,
                    "Usage: %s [--json] [--seconds N] [--no-gpu] [--no-ane]\n"
                    "          [--model PATH] [--flops N] [--shape 1,C,H,W]\n",
                    argv[0]);
            return 1;
        }
    }

    g_results = [NSMutableArray array];
    g_notes = [NSMutableArray array];

    NSDictionary *gpuInfo = doGpu ? run_gpu(seconds) : nil;
    NSDictionary *aneInfo = doAne ? run_ane(modelPath, seconds, flops, shape)
                                  : nil;

    if (asJson) {
        NSMutableDictionary *out = [@{
            @"engine": @"accel-apple",
            @"results": g_results,
            @"notes": g_notes,
        } mutableCopy];
        if (gpuInfo) out[@"gpu"] = gpuInfo;
        if (aneInfo) out[@"ane"] = aneInfo;
        NSError *e = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:out
                                                       options:0 error:&e];
        if (!json) { fprintf(stderr, "json encode failed\n"); return 1; }
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
    } else {
        printf("\n=== Accelerators (Metal / Core ML) ===\n");
        for (NSDictionary *r in g_results)
            printf("  %-30s: %14.2f %s\n",
                   [r[@"name"] UTF8String], [r[@"value"] doubleValue],
                   [r[@"unit"] UTF8String]);
        for (NSString *n in g_notes) printf("  note: %s\n", [n UTF8String]);
    }
    return 0;
} }
