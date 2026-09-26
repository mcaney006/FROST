//! frost-platform: native macOS thermal/power/memory state via NSProcessInfo.
//! Thermal *state* is an OS scheduling signal, NOT a temperature measurement.
use frost_core::ThermalState;

extern "C" {
    fn frost_thermal_state() -> i32;
    fn frost_physical_memory() -> u64;
    fn frost_low_power() -> i32;
}

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
    }
}
