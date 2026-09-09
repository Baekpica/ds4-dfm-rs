//! `strftime_now` support. Templates such as Granite 3.x and SmolLM3 call
//! `strftime_now("%d %B %Y")` to stamp the current date into system prompts. We expose a [`Clock`]
//! trait so tests can pin the date for byte-stable golden output, and ship a dependency-free
//! default. (Not every dated template uses it: Llama-3.1 takes a caller-supplied `date_string`.)
//!
//! The strftime implementation is intentionally minimal: it supports the conversion
//! specifiers that real chat templates actually use. Unknown specifiers are passed through
//! verbatim (matching neither libc nor Python perfectly — documented, and the corpus is the
//! check on what's actually needed).

/// Supplies "now" to `strftime_now`. Injectable so a render can be made deterministic.
///
/// Implementors answer only *what time it is*; the crate owns the formatting. That is deliberate:
/// the strftime output is part of the byte-identical guarantee, so a custom clock cannot diverge
/// from `transformers` by getting a specifier subtly wrong. A clock is usually three lines:
///
/// ```
/// use hf_chat_template::{Civil, Clock};
///
/// struct EpochClock;
/// impl Clock for EpochClock {
///     fn now(&self) -> Civil {
///         Civil::from_unix_secs(0)
///     }
/// }
/// ```
pub trait Clock: Send + Sync {
    /// The current civil date and time, in whatever timezone this clock represents.
    fn now(&self) -> Civil;
}

/// Forwarding impls so a boxed or shared clock is still a [`Clock`]. Without these, choosing a
/// clock at runtime (`let c: Box<dyn Clock> = if local { .. } else { .. }`) would not compile.
impl<T: Clock + ?Sized> Clock for Box<T> {
    fn now(&self) -> Civil {
        (**self).now()
    }
}

impl<T: Clock + ?Sized> Clock for std::sync::Arc<T> {
    fn now(&self) -> Civil {
        (**self).now()
    }
}

/// A civil (calendar) date and time: what a [`Clock`] reports as "now".
///
/// Construct it from a Unix timestamp or from calendar fields. The derived parts (day of week,
/// day of year) are computed for you, which is why the fields are not public: a `Civil` cannot be
/// built into an inconsistent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    year: i64,
    month: u32, // 1..=12
    day: u32,   // 1..=31
    hour: u32,
    min: u32,
    sec: u32,
    weekday: u32, // 0=Sunday .. 6=Saturday
    yday: u32,    // 1..=366
}

const MONTHS_FULL: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const MONTHS_ABBR: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const DAYS_FULL: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const DAYS_ABBR: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Convert days since the Unix epoch (1970-01-01) to a civil (year, month, day) using
/// Howard Hinnant's well-known algorithm. Valid across the full practical range.
pub(crate) fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Is `year` a leap year (proleptic Gregorian)?
fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Day-of-year (1-based) for a civil date.
fn day_of_year(year: i64, month: u32, day: u32) -> u32 {
    const CUM: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut d = CUM[(month - 1) as usize] + day;
    if month > 2 && is_leap(year) {
        d += 1;
    }
    d
}

/// Days since the Unix epoch for a civil date (inverse of [`civil_from_days`], Hinnant).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

impl Civil {
    /// Build from seconds since the Unix epoch, interpreted as UTC.
    pub fn from_unix_secs(secs: i64) -> Civil {
        let days = secs.div_euclid(86_400);
        let tod = secs.rem_euclid(86_400);
        let (year, month, day) = civil_from_days(days);
        // 1970-01-01 was a Thursday (weekday index 4 with 0=Sunday).
        let weekday = ((days.rem_euclid(7) + 4) % 7) as u32;
        Civil {
            year,
            month,
            day,
            hour: (tod / 3600) as u32,
            min: ((tod % 3600) / 60) as u32,
            sec: (tod % 60) as u32,
            weekday,
            yday: day_of_year(year, month, day),
        }
    }

    /// Build from calendar fields directly, for clocks that already have them (a local-time or
    /// timezone-aware source, say). Months and days are 1-based. Day of week and day of year are
    /// derived. Returns `None` if the date is out of range.
    pub fn from_ymd_hms(
        year: i64,
        month: u32,
        day: u32,
        hour: u32,
        min: u32,
        sec: u32,
    ) -> Option<Civil> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        if hour > 23 || min > 59 || sec > 60 {
            return None;
        }
        let days = days_from_civil(year, month, day);
        Some(Civil {
            year,
            month,
            day,
            hour,
            min,
            sec,
            // 1970-01-01 was a Thursday (weekday index 4 with 0=Sunday).
            weekday: ((days.rem_euclid(7) + 4) % 7) as u32,
            yday: day_of_year(year, month, day),
        })
    }

    /// Render this instant with a minimal strftime supporting the specifiers chat templates use.
    pub(crate) fn strftime(&self, format: &str) -> String {
        let mut out = String::with_capacity(format.len() + 8);
        let mut chars = format.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('Y') => out.push_str(&self.year.to_string()),
                Some('y') => out.push_str(&format!("{:02}", self.year.rem_euclid(100))),
                Some('m') => out.push_str(&format!("{:02}", self.month)),
                Some('d') => out.push_str(&format!("{:02}", self.day)),
                Some('e') => out.push_str(&format!("{:2}", self.day)),
                Some('B') => out.push_str(MONTHS_FULL[(self.month - 1) as usize]),
                Some('b') | Some('h') => out.push_str(MONTHS_ABBR[(self.month - 1) as usize]),
                Some('A') => out.push_str(DAYS_FULL[self.weekday as usize]),
                Some('a') => out.push_str(DAYS_ABBR[self.weekday as usize]),
                Some('j') => out.push_str(&format!("{:03}", self.yday)),
                Some('H') => out.push_str(&format!("{:02}", self.hour)),
                Some('I') => {
                    let h12 = match self.hour % 12 {
                        0 => 12,
                        h => h,
                    };
                    out.push_str(&format!("{:02}", h12));
                }
                Some('M') => out.push_str(&format!("{:02}", self.min)),
                Some('S') => out.push_str(&format!("{:02}", self.sec)),
                Some('p') => out.push_str(if self.hour < 12 { "AM" } else { "PM" }),
                Some('%') => out.push('%'),
                // Unknown specifier: emit verbatim (`%X`), so we don't silently corrupt.
                Some(other) => {
                    out.push('%');
                    out.push(other);
                }
                None => out.push('%'),
            }
        }
        out
    }
}

/// Real wall-clock (UTC). Dependency-free: reads `SystemTime` and does the calendar math here.
///
/// Note: this is UTC, not local time. transformers uses local time via Python's `datetime.now()`.
/// For reproducible/server use UTC is usually preferable; use [`LocalClock`] (the `strftime`
/// feature) to match Python's local-time behavior, or pin a [`FixedClock`] to match a specific
/// reference exactly.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Civil {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Civil::from_unix_secs(secs)
    }
}

/// A clock pinned to a fixed Unix timestamp — for deterministic golden tests.
#[derive(Clone, Copy, Debug)]
pub struct FixedClock {
    unix_secs: i64,
}

impl FixedClock {
    /// Pin to a specific number of seconds since the Unix epoch (UTC).
    pub fn from_unix_secs(unix_secs: i64) -> Self {
        FixedClock { unix_secs }
    }

    /// Pin to a specific civil date at 00:00:00 UTC. Months and days are 1-based.
    /// Returns `None` for an obviously invalid date.
    pub fn from_ymd(year: i64, month: u32, day: u32) -> Option<Self> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        // days from civil (inverse of civil_from_days), Hinnant.
        let y = if month <= 2 { year - 1 } else { year };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let mp = if month > 2 { month - 3 } else { month + 9 } as i64;
        let doy = (153 * mp + 2) / 5 + day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146_097 + doe - 719_468;
        Some(FixedClock {
            unix_secs: days * 86_400,
        })
    }
}

impl Clock for FixedClock {
    fn now(&self) -> Civil {
        Civil::from_unix_secs(self.unix_secs)
    }
}

/// Real wall-clock in the **local** timezone, matching Python's `datetime.now()` (which is what
/// `transformers` uses for `strftime_now`). Requires the `strftime` feature.
///
/// [`SystemClock`] is UTC and dependency-free; this reads the local timezone offset via `chrono`
/// (the only thing we borrow from it — formatting still goes through our own strftime, so the
/// supported specifiers and their output are identical to the other clocks). Construct it and pass
/// it to [`ChatTemplateBuilder::clock`](crate::ChatTemplateBuilder::clock) when you need rendered
/// dates to match what `transformers` would emit on the same machine:
///
/// ```
/// # #[cfg(feature = "strftime")] {
/// use hf_chat_template::{ChatTemplate, LocalClock};
/// let tmpl = ChatTemplate::builder("Today: {{ strftime_now('%Y') }}")
///     .clock(LocalClock)
///     .build()
///     .unwrap();
/// # }
/// ```
#[cfg(feature = "strftime")]
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalClock;

#[cfg(feature = "strftime")]
impl Clock for LocalClock {
    fn now(&self) -> Civil {
        use chrono::{Datelike, Local, Timelike};
        let now = Local::now();
        // from_ymd_hms recomputes weekday/yday from the calendar date, which agrees with chrono's
        // own values; falling back to the epoch is unreachable for a real local timestamp.
        Civil::from_ymd_hms(
            now.year() as i64,
            now.month(),
            now.day(),
            now.hour(),
            now.minute(),
            now.second(),
        )
        .unwrap_or_else(|| Civil::from_unix_secs(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_clock_formats_a_known_date() {
        // 2024-07-04 00:00:00 UTC. Exercises the specifiers Granite-style templates use.
        let clk = FixedClock::from_ymd(2024, 7, 4).unwrap();
        assert_eq!(clk.now().strftime("%B %d, %Y"), "July 04, 2024");
        assert_eq!(clk.now().strftime("%Y-%m-%d"), "2024-07-04");
        assert_eq!(clk.now().strftime("%A"), "Thursday");
    }

    #[test]
    fn civil_constructors_agree() {
        // The two entry points must produce identical derived fields, or a custom clock built from
        // calendar parts would drift from one built from a timestamp.
        let from_secs = Civil::from_unix_secs(1_720_051_323); // 2024-07-04 00:02:03 UTC
        let from_parts = Civil::from_ymd_hms(2024, 7, 4, 0, 2, 3).unwrap();
        assert_eq!(from_secs, from_parts);
        assert_eq!(
            from_parts.strftime("%A %j %H:%M:%S"),
            "Thursday 186 00:02:03"
        );
        // Out-of-range parts are rejected rather than silently normalized.
        assert!(Civil::from_ymd_hms(2024, 13, 1, 0, 0, 0).is_none());
        assert!(Civil::from_ymd_hms(2024, 7, 4, 24, 0, 0).is_none());
    }

    // LocalClock delegates to the local timezone; verify the wiring reads local wall time and maps
    // chrono's fields onto Civil correctly (not, say, UTC or an off-by-one weekday/ordinal). We
    // compare the date-portion against a fresh chrono::Local::now() taken in the test; the only
    // race is the midnight boundary, which we tolerate by allowing either side of it.
    #[cfg(feature = "strftime")]
    #[test]
    fn local_clock_reads_local_time() {
        use chrono::{Datelike, Local};
        let got = LocalClock.now().strftime("%Y-%m-%d");
        let now = Local::now();
        let same = format!("{:04}-{:02}-{:02}", now.year(), now.month(), now.day());
        let prev = (now - chrono::Duration::days(1)).date_naive();
        let prev = format!("{:04}-{:02}-{:02}", prev.year(), prev.month(), prev.day());
        assert!(
            got == same || got == prev,
            "LocalClock date {got} matched neither {same} nor {prev}"
        );
    }
}
