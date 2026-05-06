## Architecture

Tauri 2 desktop app: React+Vite frontend (port **1420**, HMR on **1421**) + Rust backend under `src-tauri/`. They are separate build systems — never run `cargo` to compile the frontend or `npm` for the Rust side.

- Frontend entry: `src/main.tsx` → `App.tsx`. Components live in `src/components/`, shared state in `src/context/`.
- Backend entry: `src-tauri/src/lib.rs` (library) / `main.rs` (binary). Each feature is its own module file (`gpu_processing.rs`, `raw_processing.rs`, `ai_commands.rs`, etc.).
- No test framework is configured. There are no automated tests for either side.

## Commands

```bash
npm start        # Tauri dev server (runs `tauri dev` — starts Vite + Rust in dev mode)
npm run build    # Frontend only (`vite build`) → produces dist/
npx tauri build  # Full native binary build (builds frontend first, then compiles Rust)
npm run lint     # ESLint
npm run format   # Prettier write
```

Rust checks (run from `src-tauri/`):
```bash
cargo fmt --check -p RapidRAW
cargo clippy --all-targets --all-features -- -D warnings
```

## Gotchas

- **Linux Wayland + NVIDIA** crashes: launch with `WEBKIT_DISABLE_DMABUF_RENDERER=1` or `WEBKIT_DISABLE_COMPOSITING_MODE=1`. See [#306](https://github.com/CyberTimon/RapidRAW/issues/306).
- Vite watches exclude `src-tauri/**`. Changes to Rust files require restarting the Tauri dev server.
- `rawler` is pulled from a forked GitHub repo (`CyberTimon/RapidRAW-DngLab.git`), not crates.io. Update via Cargo.toml, not `cargo update`.
- Rawler issues for unsupported camera formats should be opened against the **rawler upstream** first ([dnglab/dnglab/issues](https://github.com/dnglab/dnglab/issues)), then a RapidRAW issue to bump the dependency.
- Rust toolchain is pinned to **1.94.0** (`src-tauri/rust-toolchain.toml`). The edition is 2024.
- CI installs `libwebkit2gtk-4.1-dev` (not `-4.0`) for Linux builds — match this locally if compiling Tauri on Ubuntu.

## Style

- TypeScript: strict mode, ES2024 target, NodeNext module resolution (`tsconfig.json`).
- ESLint ignores `dist/`, `node_modules/`, `src-tauri/target/`, `src-tauri/gen/`, `src-tauri/rawler/`, `data/`. Unused vars are warnings with `_` prefix suppression.
- Prettier runs on the whole workspace. Run `npm run format:check` before committing if pre-commit hooks aren't set up.
