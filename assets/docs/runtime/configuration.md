# ADSMOD runtime configuration

Last updated: 2026-09-03

`app/resources/adsmod.json` is the only runtime value file. Its complete shape is validated by `adsmod_common.config.AdsmodConfig`; the generated `app/resources/adsmod.schema.json` is a validation aid, not a second authority.

The configuration describes one backend runtime plus the frontend and application settings. It does not select between multiple backend services. Machine learning availability is determined by whether the optional `ml` dependency extra is installed and loadable.

The same config path is used by the unified backend, launcher, maintenance scripts, and schema generation. Runtime hosts and ports, storage, database settings, public-data request policy, fitting defaults, training defaults, and polling intervals are not overridden by frontend build-time files or service-specific environment variables.

`application.datasets.allowed_extensions` is the authoritative upload policy. The backend import engine validates uploaded filenames against this configured list, the dataset capability response exposes the same list, and the frontend file picker consumes that response. The canonical v3 defaults are `.csv`, `.xls`, and `.xlsx`.

`settings/.env.example` is an optional developer-environment template for local tooling and IDEs. Copy it to `settings/.env` when needed; it does not replace or override the canonical JSON configuration.

The `runtime` section supplies the backend and frontend network settings. The `storage` section supplies the root for logs, the embedded database, checkpoints, and optional ML artifacts. Relative database paths are resolved below that root.

## Public data

`application.public_data` owns the common external-provider request policy:

- `request_timeout_seconds`: per-request timeout shared by keyless public HTTP providers;
- `retry_attempts`: bounded retry count for transient provider failures;
- `pubchem_parallel_requests`: PubChem request concurrency, constrained to a maximum of three. The provider also applies request pacing to remain below PubChem's documented five requests per second ceiling;
- `cod_max_interactive_results`: maximum COD matches ADSMOD will retrieve for one interactive search after the provider's count-first check.

`application.nist.parallel_tasks` remains specific to NIST acquisition. `application.nist.pubchem_parallel_tasks` is retained only by the existing NIST enrichment job contract and delegates to the same canonical PubChem provider implementation used by the Public Data workspace.

Provider endpoints, source identifiers, licenses, and capabilities are application-owned integration metadata rather than user configuration. See `assets/docs/architecture/public_data.md` for provider and provenance architecture.
