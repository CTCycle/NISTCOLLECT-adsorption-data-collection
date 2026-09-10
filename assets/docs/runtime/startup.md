# ADSMOD startup procedures

Last updated: 2026-09-10

## Recommended startup

```powershell
& .\start_on_windows.ps1
```

The launcher reads `app/resources/adsmod.json`, synchronizes the locked
`app/server` workspace according to the selected dependency profile, builds
the client, starts one FastAPI backend, and waits for `/health/ready` before
opening the browser. Optional machine learning support is loaded inside that
backend when its dependencies were installed.

Before starting either service, the launcher checks that the configured port
is available. If another process owns a port, startup stops with its PID and
process name; the launcher never terminates an unowned listener. After a
successful launch, use **Stop application** in the same launcher session to
stop only the backend and frontend processes started by that session.

The interactive menu is generated from structured rows. Its order is
`APPLICATION`, `SETUP & VALIDATION`, `SOURCE CONTROL` (Check before Update),
`DATA & MAINTENANCE`, and a final sequential `EXIT` option. The launcher
computes the numeric-column width from the menu size, so one- and two-digit
options remain aligned. Recursive cleanup inventories entries, removes them
deepest-first, preserves required sentinels, and reports locked or inaccessible
paths without masking the original action error.

## Source updates

Choose **Update** in the launcher menu to update the repository from
`origin/main`. The checkout must be non-detached, clean, and already on
`main`; the launcher runs `git pull --ff-only origin main` and does not switch
branches or modify local changes.

## Manual backend startup

From the repository root after `app/server/.venv` is ready:

```powershell
& .\app\server\.venv\Scripts\python.exe -m adsmod_core.cli --config .\app\resources\adsmod.json
```

This is the only backend process. If the environment was synchronized with the
`ml` extra, the process discovers and registers the optional ML extension at
startup. The Angular development server can be started from `app/client` with
`npm run dev`.

## Database startup rules

The backend runs the synchronous Alembic coordinator before serving requests.
A missing or empty SQLite file is initialized to the packaged head. A non-empty
unversioned file, an empty version table beside application tables, an unknown
revision, or a stamped-but-incomplete schema fails explicitly without schema
inference. PostgreSQL uses the same migration history and a bounded advisory
lock.

## Tests

```cmd
app\tests\run_tests.bat
```

The automated suite validates configuration, persistence, backend routes,
frontend behavior, and both dependency profiles. Live browser and
hardware-specific ML checks should be run locally on the target machine.
