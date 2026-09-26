// Real macOS platform state via Foundation NSProcessInfo.
// thermalState: 0=nominal 1=fair 2=serious 3=critical (matches NSProcessInfoThermalState).
#import <Foundation/Foundation.h>

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
