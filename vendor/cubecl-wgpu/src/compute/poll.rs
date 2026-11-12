#[cfg(not(target_family = "wasm"))]
mod _impl {
    use std::thread::JoinHandle;

    const DEFAULT_SPIN_US: u64 = 100;

    #[derive(Debug)]
    pub struct WgpuPoll {
        active_handle: std::sync::Arc<()>,
        cancel_sender: std::sync::mpsc::Sender<()>,
        poll_thread: JoinHandle<()>,
    }

    impl WgpuPoll {
        pub fn new(device: wgpu::Device) -> Self {
            let active_handle = std::sync::Arc::new(());
            let thread_check = active_handle.clone();

            let (cancel_sender, cancel_receiver) = std::sync::mpsc::channel();
            // `SPLAT_WGPU_POLL_SPIN_US`: non-blocking poll period (0 = the
            // upstream blocking wait).
            let spin_us = std::env::var("SPLAT_WGPU_POLL_SPIN_US")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(DEFAULT_SPIN_US);
            let poll_thread = std::thread::spawn(move || {
                loop {
                    // Check whether the WgpuPoll, this thread, and something else is holding
                    // a handle.
                    if std::sync::Arc::strong_count(&thread_check) > 2 {
                        // splat patch: a blocking wait for the most recent
                        // submission fires a readback's callback only once
                        // everything queued when the wait began has finished,
                        // so a reader awaiting an early copy (the lagged render
                        // counts) waits for the queue to drain and the GPU
                        // idles while the CPU encodes the next work. Polling
                        // without blocking fires each callback as soon as its
                        // own submission completes.
                        if spin_us > 0 {
                            if let Err(e) = device.poll(wgpu::PollType::Poll) {
                                log::warn!("wgpu: poll failed ({e})");
                            }
                            std::thread::sleep(std::time::Duration::from_micros(spin_us));
                            continue;
                        }
                        if let Err(e) = device.poll(wgpu::PollType::Wait {
                            submission_index: None, // Wait for most recent
                            timeout: None,
                        }) {
                            log::warn!(
                                "wgpu: requested wait timed out before the submission was completed during sync. ({e})"
                            )
                        }
                    } else {
                        // Do not cancel thread while someone still needs to poll.
                        if cancel_receiver.try_recv().is_ok() {
                            break;
                        }

                        std::thread::park();
                    }
                    std::thread::yield_now();
                }
            });

            Self {
                active_handle,
                cancel_sender,
                poll_thread,
            }
        }
        /// Get a handle, as long as it's alive the polling will be active.
        pub fn start_polling(&self) -> std::sync::Arc<()> {
            let handle = self.active_handle.clone();
            self.poll_thread.thread().unpark();
            handle
        }
    }

    impl Drop for WgpuPoll {
        fn drop(&mut self) {
            self.cancel_sender
                .send(())
                .expect("Failed to shutdown polling thread.");
            self.poll_thread.thread().unpark();
        }
    }
}

// On Wasm, the browser handles the polling loop, so we don't need anything.
#[cfg(target_family = "wasm")]
mod _impl {
    #[derive(Debug)]
    pub struct WgpuPoll {}
    impl WgpuPoll {
        pub fn new(_device: wgpu::Device) -> Self {
            Self {}
        }
        pub fn start_polling(&self) -> alloc::sync::Arc<()> {
            alloc::sync::Arc::new(())
        }
    }
}

pub(crate) use _impl::*;
