fn main() {
    // Declaring the app's commands generates `allow-<command>` permissions; a command not granted
    // in capabilities/*.json is rejected at the IPC layer.
    let attrs = tauri_build::Attributes::new()
        .app_manifest(tauri_build::AppManifest::new().commands(&["pick_data_folder", "backend_status"]));
    tauri_build::try_build(attrs).expect("tauri-build failed");
}
