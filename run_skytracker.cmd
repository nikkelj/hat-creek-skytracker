@echo off
rem Launch the native Rust skytracker from the OneDrive-safe working copy.
rem Built binary lives in CARGO_TARGET_DIR (outside OneDrive).
set SKYTRACKER_ROOT=%~dp0
set SKYTRACKER_SERIAL_TRACE=%~dp0serial_trace.log
cd /d %~dp0
C:\Users\nikke\rust-build\release\skytracker.exe %*
