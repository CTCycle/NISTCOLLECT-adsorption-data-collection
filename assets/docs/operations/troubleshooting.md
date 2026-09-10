# ADSMOD troubleshooting

Last updated: 2026-09-10

## Backend or UI unreachable

- Check the backend and frontend host/port values in `app/resources/adsmod.json`.
- Confirm the backend responds at `/health/ready`.
- Confirm the frontend preview serves the built Angular application.
- Check backend and frontend launcher logs under the configured storage root.

If frontend source, build configuration, or public assets changed after the
last build, the launcher rebuilds the bundle automatically. **Rebuild
frontend** is available when a manual rebuild is preferred.

## Configured port is already in use

The launcher reports the conflicting port, PID, and process name, then stops
without terminating that process. Close the owning application and retry, or
use **Stop application** in the launcher session that started ADSMOD. A
process left running by an earlier launcher session is intentionally treated
as an external owner because the current session cannot prove ownership of it.

## Missing dependencies

Run **Install / update dependencies** in `start_on_windows.ps1`. The installer
uses the locked backend workspace and `npm ci`. Choose the ML-enabled install
option only when training functionality is required. Resolve filesystem or
network errors and rerun the action rather than reusing a partial environment.

## Training unavailable

Query `/api/v1/system/capabilities`. If `features.machine_learning` is false,
the backend is running correctly without the optional ML extension. Re-run the
installer with machine learning dependencies enabled, then relaunch ADSMOD.
The frontend intentionally hides or blocks training routes while that
capability is unavailable.

If ML dependencies are installed but the capability remains false, inspect the
backend startup log for the recorded optional-extension load reason and verify
that the local Python and compute environment can import the installed ML
stack.

## Database startup failure

Inspect the backend logs below the configured storage root. Unknown or
non-empty unversioned schemas are rejected deliberately; export or repair them
manually instead of stamping a guessed revision. For PostgreSQL, verify
connectivity, credentials, and the role's database-creation permission.
