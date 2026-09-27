//! frost-platform: native macOS thermal/power/memory state via NSProcessInfo.
//! Thermal *state* is an OS scheduling signal, NOT a temperature measurement.
use frost_core::ThermalState;

extern "C" {
    fn frost_thermal_state() -> i32;
    fn frost_physical_memory() -> u64;
    fn frost_low_power() -> i32;
    fn frost_available_memory() -> u64;
    fn frost_memory_pressure() -> i32;
}

/// OS memory-pressure notification level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure { Normal, Warn, Critical }

/// Read the current OS thermal state. Read this before registering for change
/// notifications (the initial state is not delivered as a notification).
pub fn thermal_state() -> ThermalState {
    match unsafe { frost_thermal_state() } {
        0 => ThermalState::Nominal,
        1 => ThermalState::Fair,
        2 => ThermalState::Serious,
        _ => ThermalState::Critical,
    }
}

pub fn physical_memory_bytes() -> u64 { unsafe { frost_physical_memory() } }
pub fn low_power_mode() -> bool { unsafe { frost_low_power() != 0 } }
/// Bytes the kernel can currently hand out without paging (free+inactive+purgeable+speculative);
/// `None` when the statistics call fails.
pub fn available_memory_bytes() -> Option<u64> { let v = unsafe { frost_available_memory() }; if v == 0 { None } else { Some(v) } }
/// Latest level delivered by the libdispatch memory-pressure source (registered on first call).
pub fn memory_pressure() -> MemoryPressure {
    match unsafe { frost_memory_pressure() } { 0 => MemoryPressure::Normal, 1 => MemoryPressure::Warn, _ => MemoryPressure::Critical }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_real_platform_state() {
        let t = thermal_state();
        let mem = physical_memory_bytes();
        eprintln!("thermal={t:?} mem={:.1} GiB low_power={}", mem as f64 / 1073741824.0, low_power_mode());
        assert!(mem > 2 * 1024 * 1024 * 1024, "physical memory should be plausible");
        // thermal state is one of the four; just assert the call returns
        assert!(matches!(t, ThermalState::Nominal | ThermalState::Fair | ThermalState::Serious | ThermalState::Critical));
        let avail = available_memory_bytes().expect("vm statistics");
        eprintln!("available={:.2} GiB pressure={:?}", avail as f64 / 1073741824.0, memory_pressure());
        assert!(avail > 64 * 1024 * 1024 && avail <= mem, "available memory must be plausible: {avail}");
    }
}
