# RapidRAW — Agent Instructions

## Project Layout

```
src/              React frontend (Vite + TypeScript)
src-tauri/        Rust backend (Tauri desktop app)
  src/main.rs     Tauri entry point
data/             Test images / fixtures
```

**Single large file**: `src/App.tsx` (~214 KB). The entire React app lives here. Don't assume a typical component-per-file structure.

## Commands

| Command | What it does |
|---------|-------------|
| `npm run dev` or `npm start` | Tauri dev server (`tauri dev`) — opens the desktop app with hot reload on frontend changes |
| `npm run build` | Production Vite build |
| `npm run lint` | ESLint check |
| `npm run lint:fix` | ESLint auto-fix |
| `npm run format` / `format:check` | Prettier |

**Rust side**:
- `cargo fmt -p RapidRAW -- --check` — formatting check (CI)
- `cargo clippy --all-targets --all-features -- -D warnings` — linting (CI)

## Important Quirks

1. **No test framework.** There are no unit or integration tests configured for either the frontend or backend. You won't find Jest, Vitest, or a CI test step.
2. **GPU processing is WGSL-based.** The image editing pipeline runs on GPU via WGPU shaders — changes to adjustments/masks likely affect `src/App.tsx` and shader code. Check for `.wgsl` files in the repo if modifying rendering logic.
3. **Non-destructive edits** are stored in `.rrdata` sidecar files alongside original images.
4. **No pre-commit hooks.** Formatting is enforced only by CI (`cargo fmt`, ESLint, Prettier).
5. **Tauri CLI**: `npm run tauri` forwards to the Tauri CLI — use it for things like `tauri build`, `tauri icon`, etc.

## Development Notes

- Frontend uses NodeNext module resolution (`module: "nodenext"` in tsconfig)
- Tailwind CSS v4 via Vite plugin (not the traditional config file approach)
- VSCode recommends extensions: `rust-analyzer` and `tauri-apps.tauri-vscode`
- Linux development needs WebKit2GTK dependencies (see CI workflow for full list)
