# Phase 3A-Optional: Hot Reload - Implementation Checklist

**Spec**: [08-dspy-engine.md](./08-dspy-engine.md)
**Status**: ✅ Complete
**Target**: File watcher for automatic module reloading without server restart
**Depends On**: Phase 3A Core Engine (✅ Complete)

---

## Overview

Hot reload enables the game server to detect changes to DSPy module JSON files and automatically reload them without restarting. This is essential for:

- **Live tuning**: Adjust NPC personalities/prompts without restart
- **Development**: Iterate on prompts without restart cycle
- **A/B testing**: Swap modules to test different prompts
- **Emergency fixes**: Patch problematic responses in production

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  File System                                                │
│  modules/npc/dialogue_casual.json  [MODIFIED]               │
└─────────────────────┬───────────────────────────────────────┘
                      │ notify crate detects change
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  FileWatcher (notify crate)                                 │
│  - Watches modules/ directory recursively                   │
│  - Debounces rapid changes (100ms window)                   │
│  - Filters for .json files only                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ sends event via channel
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  HotReloadManager                                           │
│  - Receives file change events                              │
│  - Maps file paths to module IDs via manifest               │
│  - Validates hash before reload (prevents unnecessary I/O)  │
│  - Calls DSPyEngine.reload_module() or reload_all()         │
└─────────────────────┬───────────────────────────────────────┘
                      │ updates
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  DSPyEngine.modules HashMap                                 │
│  - RwLock allows concurrent reads during normal operation   │
│  - Exclusive write during reload                            │
│  - Next invoke() uses updated module                        │
└─────────────────────────────────────────────────────────────┘
```

---

## dspy-rs Compatibility Notes

**Verified against dspy-rs v0.7.3 source (`.claude/knowledge/dspy/source/`):**

### What CAN Be Hot Reloaded

1. **OptimizedModule JSON files** (our data structure, see [08-dspy-engine.md](./08-dspy-engine.md)):
   - `instruction` - System prompt text
   - `demos` - Few-shot examples
   - `metadata` - Version info, optimizer details
   - `tool_enabled` - Enable/disable tool calls

2. **How It Works**:
   - DSPyEngine stores `OptimizedModule` instances in `Arc<RwLock<HashMap>>`
   - Each `invoke()` reads module from HashMap and builds prompt from its fields
   - dspy-rs predictors (`Predict<S>`, `ChainOfThought<S>`) are stateless (`PhantomData` only)
   - Reload = re-parse JSON file → update HashMap → next invoke() uses new params

### What CANNOT Be Hot Reloaded

1. **Signature Definitions** - Compile-time Rust types (`#[derive(Signature)]`)
2. **Global LM/Adapter** - `OnceLock` in dspy-rs prevents reconfiguration after init
3. **SignatureRegistry** - Consumer registers signatures at startup, not runtime
4. **Predictor Type Changes** - `Predict` vs `ChainOfThought` is fixed per module in JSON

### Key Insight

DSPyEngine's design is **already hot-reload friendly**:
- Modules stored in `Arc<RwLock<HashMap<String, OptimizedModule>>>`
- `reload_module()` and `reload_all()` already implemented
- Each `invoke()` reads fresh module params from HashMap (no caching)
- Just need file watcher to trigger reloads automatically

---

## Prerequisites

- [x] Phase 3A Core Engine complete
- [x] `DSPyEngine::reload_all()` implemented
- [x] `DSPyEngine::reload_module()` implemented
- [x] Modules stored in `Arc<RwLock<HashMap>>`
- [x] `notify` crate added to dependencies

---

## Phase 3A-Optional Tasks

### 1. Add Dependencies

```toml
# Cargo.toml
[dependencies]
notify = "6.1"                      # Cross-platform file watcher
notify-debouncer-mini = "0.4"       # Debounce rapid changes
sha2 = "0.10"                       # Hash validation
```

- [x] Add `notify = "6.1"` to Cargo.toml
- [x] Add `notify-debouncer-mini = "0.4"` to Cargo.toml
- [x] Add `sha2 = "0.10"` to Cargo.toml
- [x] Run `cargo check` to verify dependencies

### 2. Create Hot Reload Module Structure

- [x] Create `src/inference/hotreload.rs`
- [x] Add `pub mod hotreload;` to `src/inference/mod.rs`
- [x] Export `HotReloadManager`, `HotReloadConfig`, `HotReloadEvent`, `HotReloadHandle`, `HotReloadStats` from `src/inference/mod.rs`

### 3. HotReloadConfig (`src/inference/hotreload.rs`)

- [x] Define `HotReloadConfig` struct:
  ```rust
  pub struct HotReloadConfig {
      /// Debounce window for rapid changes (default: 100ms)
      pub debounce_ms: u64,

      /// Whether to reload manifest.json changes
      pub watch_manifest: bool,

      /// File extensions to watch (default: [".json"])
      pub watch_extensions: Vec<String>,
  }
  ```
- [x] Implement `Default` for `HotReloadConfig`
- [x] Add builder methods for configuration

### 4. HotReloadEvent (`src/inference/hotreload.rs`)

- [x] Define `HotReloadEvent` enum:
  ```rust
  pub enum HotReloadEvent {
      /// Single module file changed
      ModuleChanged { module_id: String, path: PathBuf },

      /// Manifest file changed (triggers full reload)
      ManifestChanged,

      /// Module file deleted
      ModuleDeleted { module_id: String },

      /// Error during reload
      ReloadError { path: PathBuf, error: String },

      /// Module skipped (hash unchanged)
      ModuleSkipped { module_id: String },
  }
  ```

### 5. HotReloadManager (`src/inference/hotreload.rs`)

- [x] Define `HotReloadManager` struct:
  ```rust
  pub struct HotReloadManager {
      config: HotReloadConfig,
      modules_dir: PathBuf,
      modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,
      manifest: Arc<RwLock<ModuleManifest>>,
      file_hashes: Arc<RwLock<HashMap<PathBuf, String>>>,
      stats: Arc<HotReloadStats>,
      running: Arc<AtomicBool>,
      event_tx: Option<Sender<HotReloadEvent>>,
  }
  ```

- [x] Implement `HotReloadManager::new(modules, manifest, modules_dir, config)`
- [x] Implement `HotReloadManager::start()` -> Result<Receiver<HotReloadEvent>>:
  - [x] Create debounced watcher with `notify_debouncer_mini`
  - [x] Set up recursive watch on `modules_dir`
  - [x] Return event receiver for caller to monitor
- [x] Implement `HotReloadManager::stop()` - stop watching
- [x] Implement `HotReloadManager::is_running()` -> bool
- [x] Implement `HotReloadManager::stats()` -> &Arc<HotReloadStats>

### 6. File Change Handler (`src/inference/hotreload.rs`)

- [x] Implement `handle_file_event(event: DebouncedEvent, ctx: &HotReloadContext)`:
  - [x] Filter for `.json` files only
  - [x] Check if path is `manifest.json` → `ManifestChanged`
  - [x] Map file path to module_id via manifest lookup
  - [x] Determine event type (Create/Modify/Delete)
  - [x] Call appropriate reload method on engine
  - [x] Send `HotReloadEvent` to channel

- [x] Implement `path_to_module_id(path: &Path)` -> Option<String>:
  - [x] Get relative path from modules_dir
  - [x] Look up in manifest.modules by path field
  - [x] Return module_id if found

### 7. Hash Validation (Required)

- [x] Implement `compute_file_hash(path: &Path)` -> Result<String>:
  - [x] Read file contents
  - [x] Compute SHA-256 hash
  - [x] Return hex-encoded string

- [x] Implement `should_reload(path: &Path, file_hashes: &Arc<...>)` -> bool:
  - [x] Get stored hash from `file_hashes` map
  - [x] Compute current file hash
  - [x] Return true if hashes differ (or no stored hash exists)
  - [x] Update stored hash on successful reload

### 8. DSPyEngine Integration

- [x] Add `enable_hot_reload(&self, config: HotReloadConfig)` method to DSPyEngine:
  ```rust
  pub fn enable_hot_reload(&self, config: HotReloadConfig) -> Result<HotReloadHandle> {
      let mut manager = HotReloadManager::new(
          Arc::clone(&self.modules),
          Arc::clone(&self.manifest),
          self.modules_dir.clone(),
          config,
      );
      let events = manager.start()?;
      Ok(HotReloadHandle::new(manager, events))
  }
  ```

- [x] Define `HotReloadHandle` struct for managing lifetime:
  ```rust
  pub struct HotReloadHandle {
      manager: HotReloadManager,
      events: Receiver<HotReloadEvent>,
  }

  impl HotReloadHandle {
      pub fn events(&mut self) -> &mut Receiver<HotReloadEvent> { &mut self.events }
      pub fn stats(&self) -> &Arc<HotReloadStats> { self.manager.stats() }
      pub fn is_running(&self) -> bool { self.manager.is_running() }
      pub fn stop(self) { self.manager.stop(); }
  }
  ```

### 9. Error Handling

- [x] Add hot reload error variants to `DSPyEngineError`:
  ```rust
  #[error("File watcher error: {0}")]
  WatcherError(String),

  #[error("Hot reload failed for {path}: {reason}")]
  HotReloadFailed { path: String, reason: String },
  ```

- [x] Add helper methods `watcher()` and `hot_reload_failed()` to `DSPyEngineError`
- [x] Add `HotReload` variant to `ErrorKind`

### 10. Logging and Metrics

- [x] Add tracing spans for hot reload operations:
  - [x] `tracing::info!` on successful reload
  - [x] `tracing::warn!` on module deletion
  - [x] `tracing::error!` on reload failure
  - [x] `tracing::debug!` on skipped reload

- [x] Add reload counter for metrics:
  ```rust
  pub struct HotReloadStats {
      pub reloads_total: AtomicU64,
      pub reloads_failed: AtomicU64,
      pub reloads_skipped: AtomicU64,
      pub last_reload: AtomicU64, // timestamp
  }
  ```

---

## Unit Tests

### 11. Hot Reload Tests (`src/inference/hotreload.rs`)

- [x] Test: `HotReloadConfig` default values
- [x] Test: `HotReloadConfig` builder pattern
- [x] Test: `should_watch_file` filters correctly
- [x] Test: `path_to_module_id` mapping works correctly
- [x] Test: `compute_file_hash` produces consistent results
- [x] Test: `compute_file_hash` differs for different content
- [x] Test: `should_reload` returns true for no stored hash
- [x] Test: `should_reload` returns false for unchanged files (same hash)
- [x] Test: `update_hash` stores hash correctly
- [x] Test: `HotReloadStats` tracking

### 12. Integration Tests (`tests/hotreload_tests.rs`)

- [ ] Test: File watcher detects new module file
- [ ] Test: File watcher detects modified module file
- [ ] Test: File watcher detects deleted module file
- [ ] Test: Manifest change triggers full reload
- [ ] Test: Debouncing prevents multiple reloads for rapid changes
- [ ] Test: Invalid JSON doesn't crash watcher
- [ ] Test: `stop()` properly shuts down watcher
- [ ] Test: Events are sent to channel correctly

### 13. End-to-End Tests

- [ ] Test: Modify module JSON, verify next invoke() uses new instruction
- [ ] Test: Add new module to manifest, verify it becomes available
- [ ] Test: Remove module from manifest, verify it's no longer accessible

---

## Definition of Done

- [x] All unit tests pass (`cargo test`) - 10/10 passing
- [ ] All integration tests pass (file system tests - optional, deferred)
- [x] No clippy warnings (`cargo clippy`)
- [x] Hot reload can be enabled/disabled at runtime
- [x] File changes detected via debounced watcher
- [x] Debouncing prevents reload storms (100ms default)
- [x] Events channel allows monitoring reload activity
- [x] Invalid JSON files don't crash the watcher (error event sent)
- [x] Graceful shutdown of file watcher
- [x] Documentation with usage examples

---

## Files Created/Modified

```
src/
├── inference/
│   ├── mod.rs              # Added hotreload export
│   ├── hotreload.rs        # NEW: HotReloadManager, config, events, stats
│   ├── engine.rs           # Added enable_hot_reload method
│   ├── error.rs            # Added WatcherError, HotReloadFailed variants
│   └── manifest.rs         # Added Default derive to ModuleEntry
└── Cargo.toml              # Added notify, notify-debouncer-mini, sha2 dependencies
```

---

## Dependencies Added

```toml
[dependencies]
notify = "6.1"
notify-debouncer-mini = "0.4"
sha2 = "0.10"

[dev-dependencies]
tempfile = "3.19"  # For unit tests
```

---

## Usage Example

```rust
use ml_crate_dsrs::inference::{DSPyEngine, HotReloadConfig, HotReloadEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create engine as normal
    let engine = DSPyEngine::new(
        PathBuf::from("./modules"),
        adapter,
        registry,
    ).await?;

    // Enable hot reload with default config
    let mut hot_reload = engine.enable_hot_reload(HotReloadConfig::default())?;

    // Optionally monitor events in a background task
    tokio::spawn(async move {
        while let Some(event) = hot_reload.events().recv().await {
            match event {
                HotReloadEvent::ModuleChanged { module_id, .. } => {
                    println!("Module {} reloaded", module_id);
                }
                HotReloadEvent::ModuleSkipped { module_id } => {
                    println!("Module {} unchanged, skipped", module_id);
                }
                HotReloadEvent::ReloadError { path, error } => {
                    eprintln!("Reload failed for {:?}: {}", path, error);
                }
                _ => {}
            }
        }
    });

    // Use engine normally - changes will be picked up automatically
    loop {
        let result = engine.invoke("npc.dialogue", input.clone()).await?;
        // ...
    }
}
```

---

## Notes

- Hot reload is **optional** - engine works fine without it
- Use `debounce_ms: 100` to handle editors that write multiple times
- Manifest changes trigger full `reload_all()`, not incremental
- Hash validation is always enabled to prevent unnecessary reloads
- File watcher runs in separate thread, communicates via channels
- `HotReloadStats` provides metrics for monitoring reload activity

---

## Progress Log

| Date | Task | Status | Notes |
|------|------|--------|-------|
| 2025-11-25 | Phase 3A-Optional Hot Reload | ✅ Complete | All core functionality implemented, 10 unit tests passing |
| 2025-11-25 | Unit Tests Verified | ✅ Pass | `cargo test --lib hotreload` - 10/10 tests passing |
| 2025-11-25 | Clippy Clean | ✅ Pass | No warnings |
| 2025-11-25 | Integration Tests | ⏳ Deferred | File watcher real-time tests optional for future |
