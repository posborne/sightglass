# Standard Go WASI File I/O Issue

## Problem

Standard Go (GOARCH=wasm GOOS=wasip1) benchmarks fail in sightglass with the error:
```
Error reading input: open json.input: Bad file number
```

## Root Cause

The standard Go WASI runtime has a bug in preopen directory discovery. When trying to open files, it uses file descriptor `-1` (0xFFFFFFFF) instead of the correct preopen directory fd (typically `3`).

### Evidence from WASI Tracing

**TinyGo (works correctly):**
```
path_open: fd=Fd(3) ... path=*guest 0x2b550/10 ...
result=Ok(Fd(4))
```

**Standard Go (broken):**
```
fd_prestat_get: fd=Fd(3)
result=Ok(Dir(PrestatDir { pr_name_len: 1 }))  # Preopen exists!

path_open: fd=Fd(4294967295) ... path=*guest 0x40c060/10 ...  # Using -1 instead of 3!
result=Err(Error { inner: Badf })
```

The Go runtime correctly discovers the preopen at fd 3, but then fails to use it when opening files, instead passing -1 which causes EBADF (Bad file descriptor).

## Impact

- TinyGo benchmarks work perfectly
- Standard Go benchmarks fail on file I/O operations
- Cannot compare TinyGo vs Go performance in sightglass

## Workarounds Attempted

### 1. Absolute Paths
Tried using `/json.input` instead of `json.input` - **FAILED**. The same bug affects all path operations.

### 2. os.Getwd() Detection
Tried detecting the working directory with `os.Getwd()` - **FAILED**. Getwd also uses the broken preopen mechanism.

### 3. Multiple Fallback Strategies  
Tried fallback chain: relative → absolute → getwd - **FAILED**. All paths fail with the same root cause.

### 4. Direct wasmtime with Mapped Root
**SUCCESS**: `wasmtime --dir .::/` works by mapping the current directory to `/` in the guest.

However, this requires changes to the wasmtime-bench-api, which currently maps the working directory as `.` (current directory) rather than `/` (root).

**Root cause in wasmtime-bench-api**: Line 311 in `crates/bench-api/src/lib.rs`:
```rust
cx.preopened_dir(working_dir.clone(), ".", DirPerms::READ, FilePerms::READ)?;
```

Should be:
```rust
cx.preopened_dir(working_dir.clone(), "/", DirPerms::READ, FilePerms::READ)?;
```

This would allow Go's absolute path resolution to work correctly.

## Status

This is a fundamental incompatibility between:
- Go 1.21-1.25's WASI preview 1 implementation (broken preopen fd discovery)
- Wasmtime-bench-api's directory mapping strategy (maps as `.` instead of `/`)

### Possible Solutions

1. **Modify wasmtime-bench-api** to map working_dir as `/` in guest instead of `.`
2. **Wait for Go fix** - Track https://github.com/golang/go/issues/60732
3. **Use WASI preview 2** when Go supports it (wasm32-wasip2 target)
4. **Patch Go's runtime** to fix preopen fd discovery
5. **Use TinyGo exclusively** (works perfectly with current setup)

### Recommendation

For now, **use TinyGo** for Go benchmarks in sightglass. TinyGo correctly implements WASI preview 1 preopens and works flawlessly with the existing infrastructure.

##Testing Commands

Test with wasmtime directly:
```bash
cd benchmarks/go
wasmtime --dir=. --preload bench=test-bench.wasm go-json.wasm
# Error: Bad file number

wasmtime --dir=. --preload bench=test-bench.wasm tinygo-json.wasm  
# Works!
```

With WASI tracing:
```bash
WASMTIME_LOG=wasmtime_wasi=trace wasmtime --dir=. --preload bench=test-bench.wasm go-json.wasm 2>&1 | grep path_open
```

## References

- Go WASI support: https://go.dev/blog/wasi
- WASI preview 1 spec: https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md
- Related Go issues: https://github.com/golang/go/issues?q=is%3Aissue+wasi+preopen
