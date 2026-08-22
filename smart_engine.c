/*
 * smart_engine.c — read the NVMe SMART / Health log on macOS.
 *
 * Drive lifetime data (terabytes written, power-on hours, wear percentage) is
 * not available from any macOS command-line tool. `system_profiler` reports
 * only a pass/fail status, and `ioreg` exposes nothing at all — verified by
 * searching every IORegistry property for the relevant keys and finding none.
 *
 * The data lives behind an IOKit user client, which needs real code to reach.
 * That is what smartmontools does on macOS, and what this does: open the
 * NVMe SMART user client and read log page 0x02 (SMART / Health Information).
 *
 * One non-obvious detail. On Apple silicon the user client is *not* published
 * by the controller (`IONVMeController`, `AppleANS3CGv2Controller`) — both
 * return kIOReturnUnsupported. It is published by `IONVMeBlockStorageDevice`,
 * so that is what is matched. Output has been verified byte-for-byte against
 * a third-party SMART utility on the same machine.
 *
 * Needs no elevated privileges. Read-only: nothing here writes to the drive.
 *
 *   clang -O2 smart_engine.c -o smart_engine -framework IOKit \
 *         -framework CoreFoundation
 *   ./smart_engine --json
 */

#include <stdio.h>
#include <string.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <CoreFoundation/CoreFoundation.h>

/* Apple's NVMe SMART user client, as used by smartmontools. */
#define kIONVMeSMARTUserClientTypeID CFUUIDGetConstantUUIDWithBytes(NULL, \
    0xAA,0x0F,0xA6,0xF9,0xC2,0xD6,0x45,0x7F,0xB1,0x0B,0x59,0xA1,0x32,0x53,0x29,0x2F)
#define kIONVMeSMARTInterfaceID CFUUIDGetConstantUUIDWithBytes(NULL,     \
    0xCC,0xD1,0xDB,0x19,0xFD,0x9A,0x4D,0xAF,0xBF,0x95,0x12,0x45,0x4B,0x23,0x0A,0xB6)

/* Log page 0x02 is exactly 512 bytes. Kept opaque and decoded by offset so a
 * struct-packing difference between compilers cannot silently misread it. */
typedef struct { unsigned char raw[512]; } nvme_smart_log;

typedef struct IONVMeSMARTInterface {
    IUNKNOWN_C_GUTS;
    UInt16 version;
    UInt16 revision;
    IOReturn (*SMARTReadData)(void *interface, nvme_smart_log *data);
} IONVMeSMARTInterface;

/* Several counters are 128-bit little-endian. No real drive approaches 2^64
 * data units (that would be 9 zettabytes), so the low half is exact. */
static unsigned long long le64(const unsigned char *p) {
    unsigned long long v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

static void copy_property(io_service_t svc, const char *key,
                          char *out, size_t cap) {
    out[0] = '\0';
    CFStringRef name = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
    if (!name) return;
    CFTypeRef value = IORegistryEntrySearchCFProperty(
        svc, kIOServicePlane, name, NULL,
        kIORegistryIterateRecursively | kIORegistryIterateParents);
    CFRelease(name);
    if (value) {
        if (CFGetTypeID(value) == CFStringGetTypeID())
            CFStringGetCString((CFStringRef)value, out, (CFIndex)cap,
                               kCFStringEncodingUTF8);
        CFRelease(value);
    }
}

int main(int argc, char **argv) {
    int as_json = (argc > 1 && strcmp(argv[1], "--json") == 0);

    CFMutableDictionaryRef match = IOServiceMatching("IONVMeBlockStorageDevice");
    io_iterator_t it = 0;
    if (!match || IOServiceGetMatchingServices(kIOMainPortDefault, match, &it)
            != KERN_SUCCESS) {
        printf(as_json ? "{\"error\": \"no NVMe device found\"}\n"
                       : "no NVMe device found\n");
        return 1;
    }

    int found = 0;
    if (as_json) printf("{\"drives\": [");

    io_service_t svc;
    while ((svc = IOIteratorNext(it))) {
        IOCFPlugInInterface **plugin = NULL;
        SInt32 score = 0;
        if (IOCreatePlugInInterfaceForService(svc, kIONVMeSMARTUserClientTypeID,
                kIOCFPlugInInterfaceID, &plugin, &score) != KERN_SUCCESS
                || !plugin) {
            IOObjectRelease(svc);
            continue;
        }
        IONVMeSMARTInterface **smart = NULL;
        HRESULT hr = (*plugin)->QueryInterface(plugin,
            CFUUIDGetUUIDBytes(kIONVMeSMARTInterfaceID), (LPVOID *)&smart);
        if (hr != S_OK || !smart) {
            (*plugin)->Release(plugin);
            IOObjectRelease(svc);
            continue;
        }

        nvme_smart_log log;
        memset(&log, 0, sizeof(log));
        IOReturn kr = (*smart)->SMARTReadData(smart, &log);
        if (kr == kIOReturnSuccess) {
            const unsigned char *b = log.raw;
            unsigned kelvin = (unsigned)(b[1] | (b[2] << 8));
            unsigned used = b[5];
            char model[128], serial[128];
            copy_property(svc, "Model Number", model, sizeof(model));
            copy_property(svc, "Serial Number", serial, sizeof(serial));

            if (as_json) {
                printf("%s\n    {\"model\": \"%s\", \"protocol\": \"NVMe\",",
                       found ? "," : "", model);
                printf(" \"critical_warning\": %u,", b[0]);
                printf(" \"temperature_c\": %d,", (int)kelvin - 273);
                printf(" \"available_spare_pct\": %u,", b[3]);
                printf(" \"available_spare_threshold_pct\": %u,", b[4]);
                printf(" \"percentage_used\": %u,", used);
                printf(" \"data_units_read\": %llu,", le64(b + 32));
                printf(" \"data_units_written\": %llu,", le64(b + 48));
                printf(" \"host_read_commands\": %llu,", le64(b + 64));
                printf(" \"host_write_commands\": %llu,", le64(b + 80));
                printf(" \"power_cycles\": %llu,", le64(b + 112));
                printf(" \"power_on_hours\": %llu,", le64(b + 128));
                printf(" \"unsafe_shutdowns\": %llu,", le64(b + 144));
                printf(" \"media_errors\": %llu,", le64(b + 160));
                printf(" \"error_log_entries\": %llu}", le64(b + 176));
            } else {
                printf("%s\n", model[0] ? model : "NVMe drive");
                printf("  Temperature      : %d C\n", (int)kelvin - 273);
                printf("  Health           : %u%% (%u%% of rated life used)\n",
                       used > 100 ? 0 : 100 - used, used);
                printf("  Total written    : %.2f TB\n",
                       le64(b + 48) * 512000.0 / 1e12);
                printf("  Total read       : %.2f TB\n",
                       le64(b + 32) * 512000.0 / 1e12);
                printf("  Power cycles     : %llu\n", le64(b + 112));
                printf("  Power on hours   : %llu\n", le64(b + 128));
                printf("  Unsafe shutdowns : %llu\n", le64(b + 144));
                printf("  Media errors     : %llu\n", le64(b + 160));
            }
            found++;
        }
        (*smart)->Release(smart);
        (*plugin)->Release(plugin);
        IOObjectRelease(svc);
    }
    IOObjectRelease(it);

    if (as_json) printf("%s]}\n", found ? "\n  " : "");
    else if (!found) printf("no drive exposed a readable SMART log\n");
    return found ? 0 : 1;
}
