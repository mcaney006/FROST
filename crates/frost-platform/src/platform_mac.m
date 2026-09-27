// Real macOS platform state: NSProcessInfo (thermal, memory, low power) plus Mach VM statistics
// and a libdispatch memory-pressure source. Thermal state is an OS scheduling signal, not a
// temperature. All functions are cheap and safe to call from any thread.
#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <dispatch/dispatch.h>
#import <stdatomic.h>
#include <sys/sysctl.h>

// 0=nominal 1=fair 2=serious 3=critical (matches NSProcessInfoThermalState).
int frost_thermal_state(void) {
    return (int)[[NSProcessInfo processInfo] thermalState];
}

unsigned long long frost_physical_memory(void) {
    return (unsigned long long)[[NSProcessInfo processInfo] physicalMemory];
}

// low-power mode: 1 if enabled
int frost_low_power(void) {
    return [[NSProcessInfo processInfo] isLowPowerModeEnabled] ? 1 : 0;
}

// Bytes the kernel can hand out without writing anonymous memory to swap. Primary source is the
// kernel's own pressure metric, kern.memorystatus_level: the percentage of physical memory that is
// free or file-backed and pageable (droppable cache), the same number `memory_pressure` prints as
// "System-wide memory free percentage". Fallback (sysctl unavailable): free + inactive + purgeable +
// speculative pages, which undercounts on macOS because active file cache is reclaimable too.
// Returns 0 if both fail (callers treat 0 as "unknown", not as "none").
unsigned long long frost_available_memory(void) {
    int level = 0; size_t len = sizeof(level);
    if (sysctlbyname("kern.memorystatus_level", &level, &len, NULL, 0) == 0 && level > 0 && level <= 100) {
        return frost_physical_memory() / 100ULL * (unsigned long long)level;
    }
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 0;
    vm_size_t page = 0;
    if (host_page_size(host, &page) != KERN_SUCCESS || page == 0) page = 16384;
    unsigned long long pages = (unsigned long long)vm.free_count + vm.inactive_count + vm.purgeable_count + vm.speculative_count;
    return pages * (unsigned long long)page;
}

// Memory-pressure level from the OS notification source: 0 normal, 1 warn, 2 critical.
// The source is registered once on first call; the latest level is kept in an atomic.
static _Atomic int g_pressure = 0;
static dispatch_source_t g_pressure_source = NULL;
int frost_memory_pressure(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dispatch_source_t src = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
        if (src) {
            dispatch_source_set_event_handler(src, ^{
                unsigned long f = dispatch_source_get_data(src);
                int level = (f & DISPATCH_MEMORYPRESSURE_CRITICAL) ? 2 : (f & DISPATCH_MEMORYPRESSURE_WARN) ? 1 : 0;
                atomic_store(&g_pressure, level);
            });
            dispatch_resume(src);
            g_pressure_source = src; // keep it alive for the process lifetime
        }
    });
    return atomic_load(&g_pressure);
}
