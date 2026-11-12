#[cfg(target_os = "linux")]
use anyhow::Context;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs::File, process::Command};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_time: String,
    pub boot_id: String,
    pub pgid: u32,
    pub state: String,
}

pub fn identity_matches(identity: &ProcessIdentity) -> Result<bool> {
    ensure!(
        identity.pid > 1 && identity.pid <= i32::MAX as u32,
        "Refusing unsafe PID"
    );
    match process_identity(identity.pid) {
        Ok(actual) => Ok(actual.state != "Z"
            && actual.state != "exited"
            && actual.start_time == identity.start_time
            && actual.boot_id == identity.boot_id
            && actual.pgid == identity.pgid),
        Err(e)
            if e.downcast_ref::<std::io::Error>().is_some_and(|e| {
                matches!(e.kind(), std::io::ErrorKind::NotFound) || e.raw_os_error() == Some(3)
            }) =>
        {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

#[cfg(target_os = "linux")]
pub fn process_identity(pid: u32) -> Result<ProcessIdentity> {
    ensure!(pid > 1 && pid <= i32::MAX as u32, "Refusing unsafe PID");
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let (_, tail) = stat.rsplit_once(')').context("Malformed process stat")?;
    let fields: Vec<&str> = tail.split_whitespace().collect();
    ensure!(fields.len() >= 20, "Malformed process stat");
    Ok(ProcessIdentity {
        pid,
        start_time: fields[19].into(),
        pgid: fields[2].parse()?,
        state: fields[0].into(),
        boot_id: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .into(),
    })
}

#[cfg(target_os = "macos")]
pub fn process_identity(pid: u32) -> Result<ProcessIdentity> {
    ensure!(pid > 1 && pid <= i32::MAX as u32, "Refusing unsafe PID");
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of_val(&info) as i32;
    if unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDTBSDINFO,
            0,
            (&mut info as *mut libc::proc_bsdinfo).cast(),
            size,
        )
    } != size
    {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut boot: libc::timeval = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of_val(&boot);
    if unsafe {
        libc::sysctlbyname(
            c"kern.boottime".as_ptr(),
            (&mut boot as *mut libc::timeval).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(ProcessIdentity {
        pid,
        start_time: format!("{}:{}", info.pbi_start_tvsec, info.pbi_start_tvusec),
        boot_id: format!("{}:{}", boot.tv_sec, boot.tv_usec),
        pgid: info.pbi_pgid,
        state: if info.pbi_status == 5 { "Z" } else { "running" }.into(),
    })
}

#[cfg(unix)]
pub fn detach(command: &mut Command, ownership: Option<&File>, _worker: bool) -> Result<()> {
    use std::os::{fd::AsRawFd, unix::process::CommandExt};
    let fd = ownership.map(AsRawFd::as_raw_fd);
    // Only async-signal-safe syscalls may run between fork and exec.
    unsafe {
        command.pre_exec(move || {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            if let Some(fd) = fd {
                let flags = libc::fcntl(fd, libc::F_GETFD);
                if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(unix)]
pub fn interrupt(identity: &ProcessIdentity) -> Result<()> {
    ensure!(
        identity.pid > 1 && identity.pid == identity.pgid,
        "Refusing to signal a process group not owned by the job"
    );
    ensure!(
        identity_matches(identity)?,
        "Process identity changed; no signal sent"
    );
    if unsafe { libc::kill(-(identity.pgid as i32), libc::SIGINT) } < 0 {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::ESRCH) {
            return Err(error.into());
        }
    }
    Ok(())
}

#[cfg(windows)]
mod windows {
    use super::*;
    use std::{os::windows::process::CommandExt, ptr::null_mut};
    use windows_sys::Win32::{
        Foundation::*,
        System::{Console::*, JobObjects::*, SystemServices::JOB_OBJECT_QUERY, Threading::*},
    };
    pub struct JobObject(HANDLE);
    fn job_name(name: &str) -> Vec<u16> {
        format!("Local\\splat_{name}")
            .encode_utf16()
            .chain(Some(0))
            .collect()
    }
    unsafe extern "system" fn ignore_interrupt(event: u32) -> i32 {
        i32::from(event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT)
    }
    pub fn protect_supervisor_console() -> Result<()> {
        // Re-enable Ctrl-C inherited by the trainer, but keep supervisors alive to reap it.
        // Registered callbacks are reset in child processes; the ignore-Ctrl-C flag is inherited.
        ensure!(
            unsafe { SetConsoleCtrlHandler(None, 0) } != 0,
            "Enable console interrupts"
        );
        ensure!(
            unsafe { SetConsoleCtrlHandler(Some(ignore_interrupt), 1) } != 0,
            "Protect supervisor console"
        );
        Ok(())
    }
    pub fn job_has_processes(name: &str) -> Result<bool> {
        let name = job_name(name);
        let handle = unsafe { OpenJobObjectW(JOB_OBJECT_QUERY, 0, name.as_ptr()) };
        if handle.is_null() {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_FILE_NOT_FOUND as i32) {
                return Ok(false);
            }
            return Err(error.into());
        }
        let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = unsafe { std::mem::zeroed() };
        let ok = unsafe {
            QueryInformationJobObject(
                handle,
                JobObjectBasicAccountingInformation,
                (&mut info as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                std::mem::size_of_val(&info) as u32,
                null_mut(),
            )
        };
        let error = std::io::Error::last_os_error();
        unsafe {
            CloseHandle(handle);
        }
        ensure!(ok != 0, "Read Job Object membership: {error}");
        Ok(info.ActiveProcesses > 0)
    }
    impl Drop for JobObject {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
    pub fn job_object(name: &str) -> Result<JobObject> {
        let name = job_name(name);
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), name.as_ptr()) };
        ensure!(
            !handle.is_null(),
            "CreateJobObject failed: {}",
            std::io::Error::last_os_error()
        );
        // No KILL_ON_JOB_CLOSE: backend processes must survive a worker crash.
        if unsafe { AssignProcessToJobObject(handle, GetCurrentProcess()) } == 0 {
            unsafe {
                CloseHandle(handle);
            }
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(JobObject(handle))
    }
    pub fn process_identity(pid: u32) -> Result<ProcessIdentity> {
        ensure!(pid > 1 && pid <= i32::MAX as u32, "Refusing unsafe PID");
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(87) {
                return Err(std::io::Error::from(std::io::ErrorKind::NotFound).into());
            }
            return Err(e.into());
        }
        let mut created: FILETIME = unsafe { std::mem::zeroed() };
        let mut exit = created;
        let mut kernel = created;
        let mut user = created;
        let mut code = 0;
        let ok = unsafe {
            GetProcessTimes(handle, &mut created, &mut exit, &mut kernel, &mut user) != 0
                && GetExitCodeProcess(handle, &mut code) != 0
        };
        let error = std::io::Error::last_os_error();
        unsafe {
            CloseHandle(handle);
        }
        ensure!(ok, "Read process identity: {error}");
        #[repr(C)]
        struct TimeOfDay {
            boot: i64,
            current: i64,
            bias: i64,
            zone: u32,
            reserved: u32,
            boot_bias: u64,
            sleep_bias: u64,
        }
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtQuerySystemInformation(
                class: u32,
                info: *mut std::ffi::c_void,
                size: u32,
                length: *mut u32,
            ) -> i32;
        }
        let mut time: TimeOfDay = unsafe { std::mem::zeroed() };
        ensure!(
            unsafe {
                NtQuerySystemInformation(
                    3,
                    (&mut time as *mut TimeOfDay).cast(),
                    std::mem::size_of_val(&time) as u32,
                    null_mut(),
                )
            } >= 0,
            "Read Windows boot identity"
        );
        Ok(ProcessIdentity {
            pid,
            pgid: pid,
            start_time: (((created.dwHighDateTime as u64) << 32) | created.dwLowDateTime as u64)
                .to_string(),
            boot_id: time.boot.to_string(),
            state: if code == 259 { "running" } else { "exited" }.into(),
        })
    }
    pub fn detach(command: &mut Command, _ownership: Option<&File>, worker: bool) -> Result<()> {
        command.creation_flags(if worker {
            CREATE_NEW_CONSOLE
        } else {
            CREATE_NEW_PROCESS_GROUP
        });
        Ok(())
    }
    pub fn interrupt(identity: &ProcessIdentity) -> Result<()> {
        ensure!(
            identity.pid > 1 && identity.pid == identity.pgid && identity_matches(identity)?,
            "Process identity changed; no signal sent"
        );
        // Run in a short-lived helper: attaching a console must not mutate the CLI's console.
        let code = Command::new(std::env::current_exe()?)
            .arg("__splat_interrupt")
            .arg(identity.pid.to_string())
            .status()?;
        ensure!(code.success(), "Could not deliver Windows CTRL_C_EVENT");
        Ok(())
    }
    pub fn console_interrupt(pid: u32) -> Result<()> {
        ensure!(pid > 1, "Refusing unsafe PID");
        unsafe {
            FreeConsole();
        }
        ensure!(
            unsafe { AttachConsole(pid) } != 0,
            "AttachConsole: {}",
            std::io::Error::last_os_error()
        );
        unsafe {
            SetConsoleCtrlHandler(None, 1);
        }
        ensure!(
            // This is the job's private console, created by the detached worker.
            unsafe { GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0) } != 0,
            "GenerateConsoleCtrlEvent: {}",
            std::io::Error::last_os_error()
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(())
    }
}
#[cfg(windows)]
pub use windows::*;
