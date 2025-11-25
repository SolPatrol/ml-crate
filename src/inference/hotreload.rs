//! Hot Reload - File watcher for automatic module reloading
//!
//! This module provides hot reload functionality for DSPy modules,
//! allowing changes to module JSON files to be detected and applied
//! automatically without restarting the engine.
//!
//! # Architecture
//!
//! ```text
//! File System (modules/*.json)
//!     │ notify crate detects change
//!     ↓
//! FileWatcher (notify crate)
//!     │ debounced events
//!     ↓
//! HotReloadManager
//!     │ validates hash, calls reload
//!     ↓
//! DSPyEngine.modules HashMap
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ml_crate_dsrs::inference::{DSPyEngine, HotReloadConfig};
//!
//! // Enable hot reload with default config
//! let hot_reload = engine.enable_hot_reload(HotReloadConfig::default())?;
//!
//! // Monitor events in background
//! tokio::spawn(async move {
//!     while let Ok(event) = hot_reload.events().recv() {
//!         match event {
//!             HotReloadEvent::ModuleChanged { module_id, .. } => {
//!                 println!("Module {} reloaded", module_id);
//!             }
//!             _ => {}
//!         }
//!     }
//! });
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEvent};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::RwLock;

use super::error::{DSPyEngineError, Result};
use super::manifest::ModuleManifest;
use super::module::OptimizedModule;

// ============================================================================
// HotReloadContext (internal)
// ============================================================================

/// Internal context for hot reload operations
struct HotReloadContext {
    modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,
    manifest: Arc<RwLock<ModuleManifest>>,
    file_hashes: Arc<RwLock<HashMap<PathBuf, String>>>,
    modules_dir: PathBuf,
    config: HotReloadConfig,
    stats: Arc<HotReloadStats>,
    event_tx: Sender<HotReloadEvent>,
}

// ============================================================================
// HotReloadConfig
// ============================================================================

/// Configuration for hot reload functionality
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// Debounce window for rapid changes (default: 100ms)
    pub debounce_ms: u64,

    /// Whether to reload manifest.json changes (default: true)
    pub watch_manifest: bool,

    /// File extensions to watch (default: [".json"])
    pub watch_extensions: Vec<String>,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 100,
            watch_manifest: true,
            watch_extensions: vec![".json".to_string()],
        }
    }
}

impl HotReloadConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the debounce window in milliseconds
    pub fn with_debounce_ms(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Set whether to watch manifest.json changes
    pub fn with_watch_manifest(mut self, watch: bool) -> Self {
        self.watch_manifest = watch;
        self
    }

    /// Set file extensions to watch
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.watch_extensions = extensions;
        self
    }
}

// ============================================================================
// HotReloadEvent
// ============================================================================

/// Events emitted by the hot reload manager
#[derive(Debug, Clone)]
pub enum HotReloadEvent {
    /// Single module file changed and reloaded
    ModuleChanged {
        module_id: String,
        path: PathBuf,
    },

    /// Manifest file changed (triggers full reload)
    ManifestChanged,

    /// Module file deleted
    ModuleDeleted {
        module_id: String,
    },

    /// Error during reload
    ReloadError {
        path: PathBuf,
        error: String,
    },

    /// Module skipped (hash unchanged)
    ModuleSkipped {
        module_id: String,
    },
}

// ============================================================================
// HotReloadStats
// ============================================================================

/// Statistics for hot reload operations
#[derive(Debug, Default)]
pub struct HotReloadStats {
    /// Total number of successful reloads
    pub reloads_total: AtomicU64,

    /// Total number of failed reloads
    pub reloads_failed: AtomicU64,

    /// Total number of skipped reloads (hash unchanged)
    pub reloads_skipped: AtomicU64,

    /// Timestamp of last reload (Unix epoch milliseconds)
    pub last_reload: AtomicU64,
}

impl HotReloadStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful reload
    pub fn record_success(&self) {
        self.reloads_total.fetch_add(1, Ordering::Relaxed);
        self.last_reload.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            Ordering::Relaxed,
        );
    }

    /// Record a failed reload
    pub fn record_failure(&self) {
        self.reloads_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a skipped reload
    pub fn record_skip(&self) {
        self.reloads_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total reloads
    pub fn total(&self) -> u64 {
        self.reloads_total.load(Ordering::Relaxed)
    }

    /// Get failed reloads
    pub fn failed(&self) -> u64 {
        self.reloads_failed.load(Ordering::Relaxed)
    }

    /// Get skipped reloads
    pub fn skipped(&self) -> u64 {
        self.reloads_skipped.load(Ordering::Relaxed)
    }
}

// ============================================================================
// HotReloadManager
// ============================================================================

/// Manager for hot reload functionality
///
/// The manager watches the modules directory for file changes and
/// automatically reloads modules when their JSON files are modified.
pub struct HotReloadManager {
    /// Configuration
    config: HotReloadConfig,

    /// Directory containing module files
    modules_dir: PathBuf,

    /// Shared modules map (owned by DSPyEngine)
    modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,

    /// Shared manifest (owned by DSPyEngine)
    manifest: Arc<RwLock<ModuleManifest>>,

    /// File hashes for change detection
    file_hashes: Arc<RwLock<HashMap<PathBuf, String>>>,

    /// Statistics
    stats: Arc<HotReloadStats>,

    /// Running flag
    running: Arc<AtomicBool>,

    /// Event sender
    event_tx: Option<Sender<HotReloadEvent>>,
}

impl HotReloadManager {
    /// Create a new hot reload manager
    ///
    /// # Arguments
    ///
    /// * `modules` - Shared modules map from DSPyEngine
    /// * `manifest` - Shared manifest from DSPyEngine
    /// * `modules_dir` - Directory containing module files
    /// * `config` - Hot reload configuration
    pub fn new(
        modules: Arc<RwLock<HashMap<String, OptimizedModule>>>,
        manifest: Arc<RwLock<ModuleManifest>>,
        modules_dir: PathBuf,
        config: HotReloadConfig,
    ) -> Self {
        Self {
            config,
            modules_dir,
            modules,
            manifest,
            file_hashes: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(HotReloadStats::new()),
            running: Arc::new(AtomicBool::new(false)),
            event_tx: None,
        }
    }

    /// Start watching for file changes
    ///
    /// Returns a receiver for hot reload events.
    pub fn start(&mut self) -> Result<Receiver<HotReloadEvent>> {
        if self.running.load(Ordering::Relaxed) {
            return Err(DSPyEngineError::config("Hot reload already running"));
        }

        // Create event channel
        let (event_tx, event_rx) = mpsc::channel(100);
        self.event_tx = Some(event_tx.clone());

        // Create internal channel for debounced events
        let (internal_tx, mut internal_rx) = mpsc::channel::<Vec<DebouncedEvent>>(100);

        // Create debounced watcher
        let debounce_duration = Duration::from_millis(self.config.debounce_ms);

        // The notify-debouncer-mini requires a std channel
        let (std_tx, std_rx) = std::sync::mpsc::channel();

        let mut debouncer = new_debouncer(debounce_duration, std_tx)
            .map_err(|e| DSPyEngineError::watcher(format!("Failed to create debouncer: {}", e)))?;

        // Watch modules directory
        debouncer
            .watcher()
            .watch(&self.modules_dir, RecursiveMode::Recursive)
            .map_err(|e| DSPyEngineError::watcher(format!("Failed to watch directory: {}", e)))?;

        self.running.store(true, Ordering::Relaxed);
        tracing::info!("Hot reload started, watching {:?}", self.modules_dir);

        // Spawn thread to receive from std channel and forward to tokio channel
        let running = Arc::clone(&self.running);
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                match std_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(Ok(events)) => {
                        if internal_tx.blocking_send(events).is_err() {
                            break;
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("Debouncer error: {:?}", e);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            // Keep debouncer alive while thread runs
            drop(debouncer);
        });

        // Spawn task to process events
        let ctx = HotReloadContext {
            modules: Arc::clone(&self.modules),
            manifest: Arc::clone(&self.manifest),
            file_hashes: Arc::clone(&self.file_hashes),
            modules_dir: self.modules_dir.clone(),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
            event_tx: event_tx.clone(),
        };
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            while running.load(Ordering::Relaxed) {
                tokio::select! {
                    Some(events) = internal_rx.recv() => {
                        for event in events {
                            Self::handle_file_event(&event, &ctx).await;
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
            }
        });

        Ok(event_rx)
    }

    /// Stop watching for file changes
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        tracing::info!("Hot reload stopped");
    }

    /// Check if hot reload is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn stats(&self) -> &Arc<HotReloadStats> {
        &self.stats
    }

    /// Handle a file change event
    #[allow(clippy::too_many_arguments)]
    async fn handle_file_event(event: &DebouncedEvent, ctx: &HotReloadContext) {
        let path = &event.path;

        // Filter by extension
        if !Self::should_watch_file(path, &ctx.config) {
            return;
        }

        tracing::debug!("File event: {:?}", path);

        // Check if this is the manifest file
        if path.file_name().map(|n| n == "manifest.json").unwrap_or(false) {
            if ctx.config.watch_manifest {
                Self::handle_manifest_change(ctx).await;
            }
            return;
        }

        // Find module ID for this path
        let module_id = match Self::path_to_module_id(path, &ctx.manifest, &ctx.modules_dir).await {
            Some(id) => id,
            None => {
                tracing::debug!("No module ID found for path: {:?}", path);
                return;
            }
        };

        // Check if file still exists (delete event)
        if !path.exists() {
            Self::handle_module_deleted(&module_id, ctx).await;
            return;
        }

        // Check if file hash changed
        if !Self::should_reload(path, &ctx.file_hashes).await {
            ctx.stats.record_skip();
            let _ = ctx
                .event_tx
                .send(HotReloadEvent::ModuleSkipped {
                    module_id: module_id.clone(),
                })
                .await;
            tracing::debug!("Skipped reload for {} (unchanged)", module_id);
            return;
        }

        // Reload the module
        Self::handle_module_change(&module_id, path, ctx).await;
    }

    /// Check if file should be watched based on extension
    fn should_watch_file(path: &Path, config: &HotReloadConfig) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                config
                    .watch_extensions
                    .iter()
                    .any(|e| e.trim_start_matches('.') == ext)
            })
            .unwrap_or(false)
    }

    /// Map file path to module ID via manifest lookup
    async fn path_to_module_id(
        path: &Path,
        manifest: &Arc<RwLock<ModuleManifest>>,
        modules_dir: &Path,
    ) -> Option<String> {
        // Get relative path from modules_dir
        let relative_path = path.strip_prefix(modules_dir).ok()?;
        let relative_str = relative_path.to_string_lossy().replace('\\', "/");

        // Look up in manifest
        let manifest = manifest.read().await;
        for (id, entry) in &manifest.modules {
            if entry.path == relative_str {
                return Some(id.clone());
            }
        }

        None
    }

    /// Compute SHA-256 hash of file contents
    fn compute_file_hash(path: &Path) -> Result<String> {
        let contents = std::fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Check if module should be reloaded based on hash
    async fn should_reload(
        path: &Path,
        file_hashes: &Arc<RwLock<HashMap<PathBuf, String>>>,
    ) -> bool {
        let current_hash = match Self::compute_file_hash(path) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("Failed to compute hash for {:?}: {}", path, e);
                return true; // Reload on error
            }
        };

        let hashes = file_hashes.read().await;
        match hashes.get(path) {
            Some(stored_hash) => stored_hash != &current_hash,
            None => true, // No stored hash, reload
        }
    }

    /// Update stored hash for a file
    async fn update_hash(path: &Path, file_hashes: &Arc<RwLock<HashMap<PathBuf, String>>>) {
        if let Ok(hash) = Self::compute_file_hash(path) {
            file_hashes.write().await.insert(path.to_path_buf(), hash);
        }
    }

    /// Handle manifest file change
    async fn handle_manifest_change(ctx: &HotReloadContext) {
        tracing::info!("Manifest changed, reloading all modules");

        let manifest_path = ctx.modules_dir.join("manifest.json");

        // Load new manifest
        let new_manifest: ModuleManifest = match Self::load_json(&manifest_path) {
            Ok(m) => m,
            Err(e) => {
                ctx.stats.record_failure();
                let _ = ctx
                    .event_tx
                    .send(HotReloadEvent::ReloadError {
                        path: manifest_path,
                        error: e.to_string(),
                    })
                    .await;
                return;
            }
        };

        // Reload all modules
        let mut modules_guard = ctx.modules.write().await;
        modules_guard.clear();

        for (id, entry) in &new_manifest.modules {
            let module_path = ctx.modules_dir.join(&entry.path);
            match Self::load_json::<OptimizedModule>(&module_path) {
                Ok(module) => {
                    modules_guard.insert(id.clone(), module);
                }
                Err(e) => {
                    tracing::error!("Failed to load module '{}': {}", id, e);
                }
            }
        }

        *ctx.manifest.write().await = new_manifest;

        ctx.stats.record_success();
        let _ = ctx.event_tx.send(HotReloadEvent::ManifestChanged).await;
        tracing::info!("Manifest reload complete, {} modules loaded", modules_guard.len());
    }

    /// Handle module file change
    async fn handle_module_change(module_id: &str, path: &Path, ctx: &HotReloadContext) {
        tracing::info!("Reloading module: {}", module_id);

        match Self::load_json::<OptimizedModule>(path) {
            Ok(module) => {
                ctx.modules.write().await.insert(module_id.to_string(), module);
                Self::update_hash(path, &ctx.file_hashes).await;
                ctx.stats.record_success();

                let _ = ctx
                    .event_tx
                    .send(HotReloadEvent::ModuleChanged {
                        module_id: module_id.to_string(),
                        path: path.to_path_buf(),
                    })
                    .await;

                tracing::info!("Module {} reloaded successfully", module_id);
            }
            Err(e) => {
                ctx.stats.record_failure();
                let _ = ctx
                    .event_tx
                    .send(HotReloadEvent::ReloadError {
                        path: path.to_path_buf(),
                        error: e.to_string(),
                    })
                    .await;
                tracing::error!("Failed to reload module {}: {}", module_id, e);
            }
        }
    }

    /// Handle module file deletion
    async fn handle_module_deleted(module_id: &str, ctx: &HotReloadContext) {
        tracing::warn!("Module file deleted: {}", module_id);

        ctx.modules.write().await.remove(module_id);
        ctx.stats.record_success();

        let _ = ctx
            .event_tx
            .send(HotReloadEvent::ModuleDeleted {
                module_id: module_id.to_string(),
            })
            .await;
    }

    /// Load JSON file with context
    fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
        let contents = std::fs::read_to_string(path)?;
        let value = serde_json::from_str(&contents)?;
        Ok(value)
    }
}

impl std::fmt::Debug for HotReloadManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotReloadManager")
            .field("modules_dir", &self.modules_dir)
            .field("running", &self.running.load(Ordering::Relaxed))
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================
// HotReloadHandle
// ============================================================================

/// Handle for managing hot reload lifetime
///
/// The handle provides access to the event receiver and allows
/// stopping the hot reload manager.
pub struct HotReloadHandle {
    manager: HotReloadManager,
    events: Receiver<HotReloadEvent>,
}

impl HotReloadHandle {
    /// Create a new handle
    pub(crate) fn new(manager: HotReloadManager, events: Receiver<HotReloadEvent>) -> Self {
        Self { manager, events }
    }

    /// Get the event receiver
    ///
    /// Use this to monitor hot reload events in a background task.
    pub fn events(&mut self) -> &mut Receiver<HotReloadEvent> {
        &mut self.events
    }

    /// Get hot reload statistics
    pub fn stats(&self) -> &Arc<HotReloadStats> {
        self.manager.stats()
    }

    /// Check if hot reload is running
    pub fn is_running(&self) -> bool {
        self.manager.is_running()
    }

    /// Stop hot reload
    pub fn stop(self) {
        self.manager.stop();
    }
}

impl std::fmt::Debug for HotReloadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotReloadHandle")
            .field("running", &self.manager.is_running())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_reload_config_default() {
        let config = HotReloadConfig::default();
        assert_eq!(config.debounce_ms, 100);
        assert!(config.watch_manifest);
        assert_eq!(config.watch_extensions, vec![".json".to_string()]);
    }

    #[test]
    fn test_hot_reload_config_builder() {
        let config = HotReloadConfig::new()
            .with_debounce_ms(200)
            .with_watch_manifest(false)
            .with_extensions(vec![".json".to_string(), ".yaml".to_string()]);

        assert_eq!(config.debounce_ms, 200);
        assert!(!config.watch_manifest);
        assert_eq!(config.watch_extensions.len(), 2);
    }

    #[test]
    fn test_should_watch_file() {
        let config = HotReloadConfig::default();

        assert!(HotReloadManager::should_watch_file(
            Path::new("module.json"),
            &config
        ));
        assert!(!HotReloadManager::should_watch_file(
            Path::new("module.txt"),
            &config
        ));
        assert!(!HotReloadManager::should_watch_file(
            Path::new("module"),
            &config
        ));
    }

    #[test]
    fn test_compute_file_hash_consistent() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        let hash1 = HotReloadManager::compute_file_hash(file.path()).unwrap();
        let hash2 = HotReloadManager::compute_file_hash(file.path()).unwrap();

        assert_eq!(hash1, hash2);
        assert!(!hash1.is_empty());
    }

    #[test]
    fn test_compute_file_hash_differs_for_different_content() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file1 = NamedTempFile::new().unwrap();
        writeln!(file1, "content 1").unwrap();

        let mut file2 = NamedTempFile::new().unwrap();
        writeln!(file2, "content 2").unwrap();

        let hash1 = HotReloadManager::compute_file_hash(file1.path()).unwrap();
        let hash2 = HotReloadManager::compute_file_hash(file2.path()).unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_hot_reload_stats() {
        let stats = HotReloadStats::new();

        assert_eq!(stats.total(), 0);
        assert_eq!(stats.failed(), 0);
        assert_eq!(stats.skipped(), 0);

        stats.record_success();
        stats.record_success();
        stats.record_failure();
        stats.record_skip();

        assert_eq!(stats.total(), 2);
        assert_eq!(stats.failed(), 1);
        assert_eq!(stats.skipped(), 1);
    }

    #[tokio::test]
    async fn test_path_to_module_id() {
        use crate::inference::manifest::{ModuleEntry, ModuleManifest};

        let mut manifest = ModuleManifest::default();
        manifest.modules.insert(
            "npc.dialogue".to_string(),
            ModuleEntry {
                path: "npc/dialogue.json".to_string(),
                ..Default::default()
            },
        );

        let manifest = Arc::new(RwLock::new(manifest));
        let modules_dir = PathBuf::from("/modules");

        // Test exact match
        let path = PathBuf::from("/modules/npc/dialogue.json");
        let id = HotReloadManager::path_to_module_id(&path, &manifest, &modules_dir).await;
        assert_eq!(id, Some("npc.dialogue".to_string()));

        // Test non-matching path
        let path = PathBuf::from("/modules/other/module.json");
        let id = HotReloadManager::path_to_module_id(&path, &manifest, &modules_dir).await;
        assert_eq!(id, None);
    }

    #[tokio::test]
    async fn test_should_reload_no_stored_hash() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let file_hashes = Arc::new(RwLock::new(HashMap::new()));
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        // Should reload when no stored hash
        assert!(
            HotReloadManager::should_reload(file.path(), &file_hashes).await
        );
    }

    #[tokio::test]
    async fn test_should_reload_unchanged_hash() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let file_hashes = Arc::new(RwLock::new(HashMap::new()));
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        // Store the current hash
        let hash = HotReloadManager::compute_file_hash(file.path()).unwrap();
        file_hashes.write().await.insert(file.path().to_path_buf(), hash);

        // Should not reload when hash unchanged
        assert!(
            !HotReloadManager::should_reload(file.path(), &file_hashes).await
        );
    }

    #[tokio::test]
    async fn test_update_hash() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let file_hashes = Arc::new(RwLock::new(HashMap::new()));
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "test content").unwrap();

        // Update hash
        HotReloadManager::update_hash(file.path(), &file_hashes).await;

        // Hash should be stored
        let hashes = file_hashes.read().await;
        assert!(hashes.contains_key(file.path()));
    }
}
