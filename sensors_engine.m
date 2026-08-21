/*
 * sensors_engine.m — read hardware temperature sensors on Apple platforms.
 *
 * macOS exposes no public API for die temperature. `pmset -g therm` reports
 * only a throttling percentage, and `powermetrics` requires root. The sensors
 * themselves are, however, readable without privileges through the
 * IOHIDEventSystem thermal usage page — the same route `istats`, `iStat
 * Menus`, and the `Stats` app use.
 *
 * These IOHID symbols are private, so they are declared here rather than
 * imported, and every call is failure-tolerant: if a future macOS withdraws
 * them the tool reports no temperature instead of crashing.
 *
 * Deliberately a separate, tiny binary rather than part of accel_engine.m: it
 * links only Foundation and IOKit, builds in well under a second, and is still
 * available when accelerator benchmarking is skipped.
 *
 * Build:
 *   clang -O2 -fobjc-arc sensors_engine.m -o sensors_engine \
 *         -framework Foundation -framework IOKit
 * Run:
 *   ./sensors_engine --json
 */

#import <Foundation/Foundation.h>

/* --- private IOHIDEventSystem interface (see file comment) --- */
typedef struct __IOHIDEvent *IOHIDEventRef;
typedef struct __IOHIDServiceClient *IOHIDServiceClientRef;
extern CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef,
                                                CFStringRef);
extern IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef,
                                                 int64_t, int32_t, int64_t);
extern double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
extern CFTypeRef IOHIDEventSystemClientCreate(CFAllocatorRef);
extern void IOHIDEventSystemClientSetMatching(CFTypeRef, CFDictionaryRef);
extern CFArrayRef IOHIDEventSystemClientCopyServices(CFTypeRef);

#define kIOHIDEventTypeTemperature 15
#define kHIDPage_AppleVendor       0xff00
#define kHIDUsage_ThermalSensor    0x0005

/* Apple names die sensors "PMU tdie<n>" / "PMU2 tdev<n>"; the battery gauge
 * reports separately and must not be averaged into the SoC temperature. */
static BOOL is_soc_sensor(NSString *name) {
    NSString *n = name.lowercaseString;
    return ([n containsString:@"tdie"] || [n containsString:@"tdev"] ||
            [n containsString:@"soc"] || [n hasPrefix:@"pmu"]) &&
           ![n containsString:@"battery"];
}

static BOOL is_battery_sensor(NSString *name) {
    return [name.lowercaseString containsString:@"battery"];
}

int main(int argc, const char *argv[]) { @autoreleasepool {
    BOOL asJson = NO, all = NO;
    for (int i = 1; i < argc; ++i) {
        NSString *a = @(argv[i]);
        if ([a isEqualToString:@"--json"]) asJson = YES;
        else if ([a isEqualToString:@"--all"]) all = YES;
    }

    CFTypeRef system = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
    if (!system) {
        if (asJson) printf("{\"error\":\"cannot create IOHID client\"}\n");
        else fprintf(stderr, "cannot create IOHID client\n");
        return 1;
    }
    NSDictionary *match = @{@"PrimaryUsagePage": @(kHIDPage_AppleVendor),
                            @"PrimaryUsage": @(kHIDUsage_ThermalSensor)};
    IOHIDEventSystemClientSetMatching(system,
                                      (__bridge CFDictionaryRef)match);

    CFArrayRef services = IOHIDEventSystemClientCopyServices(system);
    if (!services) {
        if (asJson) printf("{\"error\":\"no thermal services\"}\n");
        return 1;
    }

    NSMutableArray *sensors = [NSMutableArray array];
    double socSum = 0, socMax = -1e9, battery = -1e9;
    int socCount = 0;

    long count = CFArrayGetCount(services);
    for (long i = 0; i < count; ++i) {
        IOHIDServiceClientRef svc =
            (IOHIDServiceClientRef)CFArrayGetValueAtIndex(services, i);
        CFTypeRef nameRef = IOHIDServiceClientCopyProperty(svc,
                                                           CFSTR("Product"));
        if (!nameRef) continue;
        NSString *name = (__bridge_transfer NSString *)nameRef;

        IOHIDEventRef event =
            IOHIDServiceClientCopyEvent(svc, kIOHIDEventTypeTemperature, 0, 0);
        if (!event) continue;
        double celsius =
            IOHIDEventGetFloatValue(event,
                                    kIOHIDEventTypeTemperature << 16);
        /* Implausible readings mean the sensor is unpopulated. */
        if (celsius <= -50.0 || celsius >= 150.0) continue;

        if (is_battery_sensor(name)) {
            battery = celsius;
        } else if (is_soc_sensor(name)) {
            socSum += celsius;
            if (celsius > socMax) socMax = celsius;
            ++socCount;
        }
        if (all)
            [sensors addObject:@{@"name": name, @"celsius": @(celsius)}];
    }
    CFRelease(services);

    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    if (socCount) {
        out[@"cpu_celsius"] = @(socMax);              /* hottest die sensor */
        out[@"cpu_avg_celsius"] = @(socSum / socCount);
        out[@"sensor_count"] = @(socCount);
    }
    if (battery > -1e9) out[@"battery_celsius"] = @(battery);
    if (all) out[@"sensors"] = sensors;

    if (asJson) {
        NSError *e = nil;
        NSData *j = [NSJSONSerialization dataWithJSONObject:out options:0
                                                      error:&e];
        if (!j) { printf("{\"error\":\"encode failed\"}\n"); return 1; }
        fwrite(j.bytes, 1, j.length, stdout);
        fputc('\n', stdout);
    } else {
        if (out[@"cpu_celsius"])
            printf("CPU  %.1f C (max of %d sensors, avg %.1f C)\n",
                   [out[@"cpu_celsius"] doubleValue],
                   [out[@"sensor_count"] intValue],
                   [out[@"cpu_avg_celsius"] doubleValue]);
        if (out[@"battery_celsius"])
            printf("Batt %.1f C\n", [out[@"battery_celsius"] doubleValue]);
        for (NSDictionary *s in sensors)
            printf("  %-24s %6.2f C\n", [s[@"name"] UTF8String],
                   [s[@"celsius"] doubleValue]);
    }
    return 0;
} }
