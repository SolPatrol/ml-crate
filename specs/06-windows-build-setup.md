# Windows Build Setup for llama-cpp-2

## Prerequisites

Building `llama-cpp-2` on Windows requires the following tools:

### 1. Visual Studio 2022 Build Tools
Install with C++ workload:
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools
```
Then open Visual Studio Installer and add "Desktop development with C++" workload.

### 2. CMake (v4.0+)
```powershell
winget install Kitware.CMake
```

### 3. Ninja Build System
```powershell
winget install Ninja-build.Ninja
```

### 4. LLVM/Clang (for bindgen)
```powershell
winget install LLVM.LLVM
```

### 5. Vulkan SDK (for Vulkan backend)
```powershell
winget install KhronosGroup.VulkanSDK
```

## Cargo Configuration

The project includes `.cargo/config.toml` with required settings:

```toml
[env]
# Vulkan SDK path
VULKAN_SDK = "C:\\VulkanSDK\\1.4.328.1"

# CMake and Ninja paths
CMAKE = "C:\\Program Files\\CMake\\bin\\cmake.exe"
CMAKE_GENERATOR = "Ninja"
CMAKE_MAKE_PROGRAM = "C:\\Users\\<USER>\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\\ninja.exe"

# PATH extension for nested cmake calls
PATH = { value = "...ninja path...;...cmake path...;${PATH}", relative = false }

# Shorter target directory to avoid Windows path length issues
[build]
target-dir = "C:\\cb"
```

**Important**: Update `<USER>` placeholders with your Windows username.

## Why These Settings?

1. **Ninja generator**: VS generator has issues with ExternalProject in nested builds
2. **Short target-dir**: Windows has 260-character path limits; nested cmake builds can exceed this
3. **PATH extension**: Nested cmake ExternalProject calls need Ninja/CMake in PATH

## Build Commands

### Default (Vulkan) - AMD, NVIDIA, Intel GPUs
```powershell
cargo build
cargo check
```

### CUDA - NVIDIA GPUs (+10-20% performance vs Vulkan)
```powershell
cargo build --no-default-features --features cuda
```

### CPU only (no GPU acceleration)
```powershell
cargo build --no-default-features --features cpu
```

## Troubleshooting

### Error: "cmake not found"
Ensure CMake is installed and `CMAKE` env var points to `cmake.exe`.

### Error: "Ninja not found" / "CMAKE_MAKE_PROGRAM is not set"
Ensure Ninja is installed and both `CMAKE_MAKE_PROGRAM` and `PATH` include the ninja directory.

### Error: "LNK1104: cannot open file '...intermediate.manifest'"
This is a Windows path length issue. Ensure `target-dir = "C:\\cb"` is set in `.cargo/config.toml`.

### Error: "Visual studio version detected but this crate doesn't know how to generate cmake files"
You may have VS Insiders installed. Uninstall it and use VS 2022 (stable):
```powershell
winget uninstall Microsoft.VisualStudio.Community.Insiders
winget install Microsoft.VisualStudio.2022.BuildTools
```

### Error: "libclang not found"
Set `LIBCLANG_PATH`:
```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
```

## Shell Restart

After installing tools via winget, restart your terminal/IDE for PATH changes to take effect.
