//! Filter wheels: the ASI electronic filter wheel (EFW) on the main scope
//! through ZWO's `EFW_filter.dll` (loaded at runtime with libloading — the
//! same pattern as the camera SDK; the app degrades gracefully when the DLL
//! or the wheel is absent) and manual wheels whose position the operator
//! reports. Filter assignments (slot -> name) live in config.json
//! `filter_wheels` and are edited / persisted from the Cameras screen.

use crate::state::{FilterWheelSnapshot, FwCmd, Shared, WheelState};
use std::os::raw::{c_char, c_int};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[repr(C)]
#[allow(non_snake_case)]
struct EfwInfo {
    ID: c_int,
    Name: [c_char; 64],
    slotNum: c_int,
}

/// Minimal EFW SDK surface (ZWO EFW_filter.dll).
struct EfwSdk {
    _lib: libloading::Library,
    get_num: unsafe extern "C" fn() -> c_int,
    get_id: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    open: unsafe extern "C" fn(c_int) -> c_int,
    get_property: unsafe extern "C" fn(c_int, *mut EfwInfo) -> c_int,
    get_position: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    set_position: unsafe extern "C" fn(c_int, c_int) -> c_int,
    close: unsafe extern "C" fn(c_int) -> c_int,
}

impl EfwSdk {
    fn load(path: &str) -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new(path).map_err(|e| format!("{path}: {e}"))?;
            macro_rules! sym {
                ($name:literal, $t:ty) => {
                    *lib.get::<$t>($name).map_err(|e| format!("{}: {e}", String::from_utf8_lossy($name)))?
                };
            }
            let get_num = sym!(b"EFWGetNum\0", unsafe extern "C" fn() -> c_int);
            let get_id = sym!(b"EFWGetID\0", unsafe extern "C" fn(c_int, *mut c_int) -> c_int);
            let open = sym!(b"EFWOpen\0", unsafe extern "C" fn(c_int) -> c_int);
            let get_property = sym!(b"EFWGetProperty\0", unsafe extern "C" fn(c_int, *mut EfwInfo) -> c_int);
            let get_position = sym!(b"EFWGetPosition\0", unsafe extern "C" fn(c_int, *mut c_int) -> c_int);
            let set_position = sym!(b"EFWSetPosition\0", unsafe extern "C" fn(c_int, c_int) -> c_int);
            let close = sym!(b"EFWClose\0", unsafe extern "C" fn(c_int) -> c_int);
            Ok(EfwSdk { _lib: lib, get_num, get_id, open, get_property, get_position, set_position, close })
        }
    }
    fn num(&self) -> i32 {
        unsafe { (self.get_num)() }
    }
    fn id(&self, index: i32) -> Option<i32> {
        let mut id = 0;
        (unsafe { (self.get_id)(index, &mut id) } == 0).then_some(id)
    }
    fn open(&self, id: i32) -> bool {
        unsafe { (self.open)(id) == 0 }
    }
    fn property(&self, id: i32) -> Option<(String, i32)> {
        let mut info = EfwInfo { ID: 0, Name: [0; 64], slotNum: 0 };
        if unsafe { (self.get_property)(id, &mut info) } != 0 {
            return None;
        }
        let bytes: Vec<u8> = info.Name.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        Some((String::from_utf8_lossy(&bytes).to_string(), info.slotNum))
    }
    fn position(&self, id: i32) -> Option<i32> {
        let mut p = -1;
        (unsafe { (self.get_position)(id, &mut p) } == 0).then_some(p)
    }
    fn set_position(&self, id: i32, pos: i32) -> bool {
        unsafe { (self.set_position)(id, pos) == 0 }
    }
    fn close(&self, id: i32) {
        unsafe {
            (self.close)(id);
        }
    }
}

pub fn spawn(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<FwCmd>, repo_root: std::path::PathBuf) {
    std::thread::Builder::new().name("filter-wheels".into()).spawn(move || run(shared, rx, repo_root)).expect("spawn filter wheel worker");
}

fn run(shared: Arc<Shared>, rx: crossbeam_channel::Receiver<FwCmd>, root: std::path::PathBuf) {
    let cfg = shared.config.clone();
    let dll = if std::path::Path::new(&cfg.efw_dll).is_absolute() { cfg.efw_dll.clone() } else { root.join(&cfg.efw_dll).to_string_lossy().to_string() };
    let sdk = EfwSdk::load(&dll);
    let sdk_status = match &sdk {
        Ok(s) => format!("EFW SDK loaded, {} wheel(s) found", s.num()),
        Err(e) => format!("EFW SDK unavailable ({e}) -- electronic wheel shown as offline"),
    };
    let sdk = sdk.ok();

    let mut wheels: Vec<WheelState> = cfg
        .filter_wheels
        .iter()
        .map(|w| WheelState {
            name: w.name.clone(),
            kind: w.kind.clone(),
            camera: w.camera,
            slots: w.slots.clone(),
            position: w.position.min(w.slots.len().saturating_sub(1)),
            target: None,
            moving: false,
            connected: w.kind != "efw",
            message: if w.kind == "efw" { "offline".into() } else { "manual".into() },
        })
        .collect();
    // Open EFW wheels.
    let mut efw_ids: Vec<Option<i32>> = vec![None; wheels.len()];
    if let Some(s) = &sdk {
        for (i, w) in cfg.filter_wheels.iter().enumerate() {
            if w.kind != "efw" {
                continue;
            }
            if let Some(id) = s.id(w.efw_index as i32) {
                if s.open(id) {
                    if let Some((name, slots)) = s.property(id) {
                        wheels[i].connected = true;
                        wheels[i].message = format!("{name} · {slots} slots");
                        if wheels[i].slots.len() < slots as usize {
                            wheels[i].slots.resize(slots as usize, String::new());
                        }
                        efw_ids[i] = Some(id);
                    }
                }
            }
        }
    }
    let mut dirty = false;
    let mut last_poll = Instant::now() - Duration::from_secs(5);
    let publish = |wheels: &Vec<WheelState>, sdk_status: &str| {
        shared.filter_wheels.store(Arc::new(FilterWheelSnapshot { wheels: wheels.clone(), sdk: sdk_status.to_string() }));
    };
    publish(&wheels, &sdk_status);

    loop {
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                FwCmd::Goto { wheel, slot } => {
                    if let Some(w) = wheels.get_mut(wheel) {
                        if slot < w.slots.len() {
                            if let (Some(s), Some(id)) = (&sdk, efw_ids[wheel]) {
                                if s.set_position(id, slot as i32) {
                                    w.target = Some(slot);
                                    w.moving = true;
                                    w.message = format!("moving to {}", slot + 1);
                                } else {
                                    w.message = "EFWSetPosition failed".into();
                                }
                            } else if w.kind != "efw" {
                                w.position = slot;
                                dirty = true;
                            } else {
                                w.message = "wheel offline".into();
                            }
                        }
                    }
                }
                FwCmd::MarkPosition { wheel, slot } => {
                    if let Some(w) = wheels.get_mut(wheel) {
                        if slot < w.slots.len() {
                            w.position = slot;
                            dirty = true;
                        }
                    }
                }
                FwCmd::SetSlotName { wheel, slot, name } => {
                    if let Some(w) = wheels.get_mut(wheel) {
                        if slot < w.slots.len() {
                            w.slots[slot] = name;
                            dirty = true;
                        }
                    }
                }
                FwCmd::Save => dirty = true,
            }
        }
        // Poll EFW positions at 4 Hz.
        if last_poll.elapsed() > Duration::from_millis(250) {
            last_poll = Instant::now();
            if let Some(s) = &sdk {
                for (i, id) in efw_ids.iter().enumerate() {
                    let Some(id) = *id else { continue };
                    match s.position(id) {
                        Some(p) if p >= 0 => {
                            let w = &mut wheels[i];
                            let was_moving = w.moving;
                            w.moving = false;
                            if w.position != p as usize || was_moving {
                                w.position = p as usize;
                                w.target = None;
                                w.message = format!("at {}", p + 1);
                                dirty = true;
                            }
                        }
                        Some(_) => wheels[i].moving = true,
                        None => {
                            wheels[i].connected = false;
                            wheels[i].message = "lost".into();
                        }
                    }
                }
            }
        }
        if dirty {
            dirty = false;
            persist(&cfg.path, &wheels);
        }
        publish(&wheels, &sdk_status);
        std::thread::sleep(Duration::from_millis(100));
    }
    #[allow(unreachable_code)]
    {
        if let Some(s) = &sdk {
            for id in efw_ids.iter().flatten() {
                s.close(*id);
            }
        }
    }
}

/// Write filter assignments / positions back into config.json (only the
/// `filter_wheels` key; everything else untouched).
fn persist(path: &std::path::Path, wheels: &[WheelState]) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let Ok(mut raw) = serde_json::from_str::<serde_json::Value>(&text) else { return };
    let Some(o) = raw.as_object_mut() else { return };
    let existing = o.get("filter_wheels").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let arr: Vec<serde_json::Value> = wheels
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let mut v = existing.get(i).cloned().unwrap_or_else(|| serde_json::json!({}));
            let m = v.as_object_mut().unwrap();
            m.insert("name".into(), serde_json::json!(w.name));
            m.insert("kind".into(), serde_json::json!(w.kind));
            m.insert("camera".into(), serde_json::json!(w.camera));
            m.entry("efw_index").or_insert(serde_json::json!(0));
            m.insert("slots".into(), serde_json::json!(w.slots));
            m.insert("position".into(), serde_json::json!(w.position));
            v
        })
        .collect();
    o.insert("filter_wheels".into(), serde_json::Value::Array(arr));
    if let Ok(s) = serde_json::to_string_pretty(&raw) {
        let _ = std::fs::write(path, s);
    }
}
