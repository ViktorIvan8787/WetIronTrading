mod ai;
mod engine;
mod kronos;
mod shared;
mod ui;
mod viewer;
mod yahoo;

/// `wetiron` opens the window (starting the engine if needed); `wetiron engine` is the background engine.
fn main() -> std::io::Result<()> {
    // the window closes on a crash, so keep the reason: %LOCALAPPDATA%\WetIronTrading\crash.log
    std::panic::set_hook(Box::new(|info| {
        let msg = format!("{} {:?}: {info}\n{}\n", chrono::Local::now(), std::env::args().collect::<Vec<_>>(), std::backtrace::Backtrace::force_capture());
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(shared::data_dir().join("crash.log")).and_then(|mut f| std::io::Write::write_all(&mut f, msg.as_bytes()));
    }));
    if std::env::args().nth(1).as_deref() == Some("engine") {
        engine::run();
        Ok(())
    } else {
        let r = viewer::run();
        if let Err(e) = &r {
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(shared::data_dir().join("crash.log")).and_then(|mut f| std::io::Write::write_all(&mut f, format!("{} window error: {e}\n", chrono::Local::now()).as_bytes()));
        }
        r
    }
}
