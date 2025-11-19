# CUDA Development Setup on Windows

This project uses Candle with CUDA support, which requires additional setup on Windows.

## Problem: `cl.exe not found` Error

When running `cargo clippy` or `cargo build`, you might see:

```
nvcc fatal: Cannot find compiler 'cl.exe' in PATH
```

This happens because NVCC (NVIDIA's CUDA compiler) needs Microsoft's C++ compiler (`cl.exe`) from Visual Studio.

## Solution

You have **three options** to fix this:

---

### Option 1: Use Developer Command Prompt (Easiest)

1. Open **Start Menu**
2. Search for **"x64 Native Tools Command Prompt for VS 2022"**
3. Run it (opens a terminal with PATH already configured)
4. Navigate to your project:
   ```cmd
   cd C:\Projects\con-ai\ml-crate-dsrs
   ```
5. Run cargo commands normally:
   ```cmd
   cargo clippy
   cargo build
   cargo test
   ```

**Pros**: No manual setup needed
**Cons**: Must remember to use the special terminal

---

### Option 2: Run Setup Script (Recommended)

We've created setup scripts that configure your current terminal:

#### Using PowerShell (Recommended):

```powershell
# Run this ONCE per terminal session
. .\setup_cuda_env.ps1

# Then run cargo commands
cargo clippy
```

#### Using Command Prompt:

```cmd
REM Run this ONCE per terminal session
setup_cuda_env.bat

REM Then run cargo commands
cargo clippy
```

**What the scripts do**:
- Set `NVCC_CCBIN` to point to Visual Studio's cl.exe
- Add cl.exe directory to PATH
- Verify cl.exe is accessible

**Pros**: Works in any terminal after sourcing
**Cons**: Must run every time you open a new terminal

---

### Option 3: Permanent Environment Variable (Advanced)

Set a **system-wide** environment variable:

1. Open **System Properties** → **Environment Variables**
2. Under **System variables**, click **New**
3. Set:
   - **Variable name**: `NVCC_CCBIN`
   - **Variable value**: `C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`
4. Edit **Path** variable, add new entry:
   - `C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`
5. Click **OK** to close all dialogs
6. **Restart your terminal** (or reboot)

**Pros**: Works automatically in all terminals forever
**Cons**: Requires admin access, affects all projects

---

## Verification

After setting up, verify cl.exe is found:

### PowerShell:
```powershell
Get-Command cl.exe
```

### Command Prompt:
```cmd
where cl.exe
```

### Git Bash:
```bash
which cl.exe
```

Should show the path to cl.exe. If found, you're ready to go!

---

## Run Clippy

Once cl.exe is in PATH:

```bash
# Full check (compiles CUDA kernels)
cargo clippy --all-targets -- -D warnings

# Faster check (lib only, may skip some CUDA checks)
cargo clippy --lib -- -D warnings
```

---

## Notes

- Your Visual Studio version: **2022 Insiders (v18)**
- MSVC version: **14.44.35207**
- Architecture: **x64 (64-bit)**

If Visual Studio updates and the path changes, you'll need to update the paths in:
- `setup_cuda_env.bat`
- `setup_cuda_env.ps1`
- Or your system environment variables

---

## Alternative: Skip CUDA Compilation

If you don't need CUDA features during development, you can disable them:

```bash
# Build without CUDA (CPU only)
cargo build --no-default-features

# Or set in Cargo.toml temporarily
[dependencies]
candle-core = { version = "0.9.1", default-features = false }
```

**Warning**: This disables GPU acceleration. Only use for quick syntax checks.

---

## Troubleshooting

### "cl.exe still not found after setup"

Check if the path is correct:
```bash
ls "/c/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
```

If file doesn't exist, find the correct path:
```bash
find "/c/Program Files/Microsoft Visual Studio" -name "cl.exe" 2>/dev/null | head -1
```

Update the scripts with the correct path.

### "Access denied" when setting environment variables

Run PowerShell/CMD as **Administrator**.

### Different Visual Studio version

If you have a different Visual Studio version (e.g., Community instead of Insiders), adjust the path:
- Insiders → `18/Insiders`
- Community → `2022/Community`
- Professional → `2022/Professional`
- Enterprise → `2022/Enterprise`

---

## Quick Reference

| What | Command |
|------|---------|
| Setup (PowerShell) | `. .\setup_cuda_env.ps1` |
| Setup (CMD) | `setup_cuda_env.bat` |
| Verify | `where cl.exe` or `Get-Command cl.exe` |
| Run clippy | `cargo clippy --all-targets -- -D warnings` |
| Run tests | `cargo test` |
| Run build | `cargo build --release` |

---

**Status**: Environment setup documented
**Last Updated**: 2025-11-18
