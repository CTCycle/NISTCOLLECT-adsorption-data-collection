import { Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink, RouterLinkActive } from '@angular/router';
import type {
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResult,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialPageResponse,
    PublicDataView,
    PublicSourceSummary,
    StructurePageResponse,
    StructureRecordView,
} from '../../models/public-data.model';
import {
    fetchPublicAdsorption,
    fetchPublicAdsorptionDetail,
    fetchPublicChemicalDetail,
    fetchPublicChemicals,
    fetchPublicMaterials,
    fetchPublicSources,
    fetchPublicStructureDetail,
    fetchPublicStructures,
    importCOD,
    resolvePubChem,
    searchCOD,
} from '../../services/public-data.service';
import { NistCollectionRowsComponent } from '../nist/nist-collection-rows.component';

const VIEWS: readonly PublicDataView[] = [
    'overview',
    'adsorption',
    'materials',
    'chemicals',
    'structures',
    'sources',
];

@Component({
    selector: 'adsmod-public-data-workspace',
    standalone: true,
    imports: [FormsModule, RouterLink, RouterLinkActive, NistCollectionRowsComponent],
    template: `
        <div class="public-workspace">
            <nav class="public-tabs" aria-label="Public data workspace">
                @for (tab of views; track tab) {
                    <a [routerLink]="['/public-data', tab]" routerLinkActive="active">{{ viewLabel(tab) }}</a>
                }
            </nav>

            @if (error()) {
                <div class="public-alert error" role="alert">
                    <strong>Public data request failed.</strong>
                    <span>{{ error() }}</span>
                    <button type="button" class="button secondary" (click)="reload()">Retry</button>
                </div>
            }

            @if (loading()) {
                <div class="public-loading" aria-live="polite">
                    <span class="public-spinner" aria-hidden="true"></span>
                    <span>Loading {{ viewLabel(view()) }}…</span>
                </div>
            }

            @switch (view()) {
                @case ('overview') {
                    <section class="public-intro">
                        <div>
                            <p class="eyebrow">Normalized public workspace</p>
                            <h2>Adsorption, chemistry, and structures in one provenance-aware model</h2>
                            <p>Discover external records, cache normalized scientific metadata locally, and retain source identifiers, retrieval times, original units, and source links.</p>
                        </div>
                        <div class="overview-totals" aria-label="Local public data counts">
                            <div><strong>{{ overviewAdsorptionTotal() }}</strong><span>adsorption records</span></div>
                            <div><strong>{{ overviewMaterialTotal() }}</strong><span>materials</span></div>
                            <div><strong>{{ overviewChemicalTotal() }}</strong><span>chemicals</span></div>
                            <div><strong>{{ overviewStructureTotal() }}</strong><span>structures</span></div>
                        </div>
                    </section>
                    <section class="public-section">
                        <div class="section-heading">
                            <div><p class="eyebrow">Provider coverage</p><h3>Integrated sources</h3></div>
                            <a class="text-link" routerLink="/public-data/sources">Inspect sources</a>
                        </div>
                        <div class="source-summary-grid">
                            @for (source of sources(); track source.key) {
                                <article class="source-summary">
                                    <div class="source-summary-head">
                                        <div><strong>{{ source.name }}</strong><span class="source-key">{{ source.key }}</span></div>
                                        <span class="source-status" [class.available]="source.status === 'available'" [class.unavailable]="source.status === 'unavailable'">{{ source.status }}</span>
                                    </div>
                                    <p>{{ source.description }}</p>
                                    <div class="capability-list">
                                        @for (capability of source.capabilities; track capability) { <span>{{ capability }}</span> }
                                    </div>
                                    <div class="source-summary-foot"><span>{{ source.record_count }} local source records</span><a [href]="source.homepage_url" target="_blank" rel="noopener noreferrer">Open source</a></div>
                                </article>
                            }
                        </div>
                    </section>
                }
                @case ('adsorption') {
                    <section class="public-section">
                        <div class="section-heading"><div><p class="eyebrow">Normalized measurements</p><h3>Adsorption data</h3><p>Filter locally cached isotherms without loading complete datasets into the browser.</p></div></div>
                        <div class="filter-bar">
                            <label>Source<select [(ngModel)]="adsorptionSource"><option value="">All sources</option>@for (source of sources(); track source.key) {<option [value]="source.key">{{ source.name }}</option>}</select></label>
                            <label>Material<input [(ngModel)]="adsorptionMaterial" placeholder="e.g. carbon, MOF-5" /></label>
                            <label>Adsorbate<input [(ngModel)]="adsorptionAdsorbate" placeholder="e.g. methane" /></label>
                            <label>Min temperature (K)<input [(ngModel)]="temperatureMin" type="number" min="0" /></label>
                            <label>Max temperature (K)<input [(ngModel)]="temperatureMax" type="number" min="0" /></label>
                            <div class="filter-actions"><button class="button primary" type="button" (click)="applyAdsorptionFilters()">Apply</button><button class="button secondary" type="button" (click)="resetAdsorptionFilters()">Reset</button></div>
                        </div>
                        <div class="table-frame">
                            <table>
                                <thead><tr><th>Source</th><th>Source ID</th><th>Material</th><th>Adsorbate</th><th>Temperature</th><th>Pressure range</th><th>Uptake range</th><th>Points</th><th></th></tr></thead>
                                <tbody>
                                    @for (item of adsorption()?.items ?? []; track item.id) {
                                        <tr>
                                            <td><span class="source-key">{{ item.source }}</span></td>
                                            <td class="truncate-cell" [title]="item.external_id">{{ item.external_id }}</td>
                                            <td class="truncate-cell" [title]="item.material">{{ item.material }}</td>
                                            <td class="truncate-cell" [title]="item.adsorbates.join(', ')">{{ item.adsorbates.join(', ') }}</td>
                                            <td>{{ formatNumber(item.temperature_k) }} K</td>
                                            <td>{{ range(item.pressure_min_pa, item.pressure_max_pa, 'Pa') }}</td>
                                            <td>{{ range(item.uptake_min_mol_kg, item.uptake_max_mol_kg, 'mol/kg') }}</td>
                                            <td>{{ item.point_count }}</td>
                                            <td><button class="row-action" type="button" (click)="openAdsorption(item.id)">Inspect</button></td>
                                        </tr>
                                    } @empty {
                                        <tr><td colspan="9" class="empty-table">No adsorption records match the current filters.</td></tr>
                                    }
                                </tbody>
                            </table>
                        </div>
                        <div class="pagination-row">{{ pageSummary(adsorption()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(adsorption()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (adsorptionDetail(); as detail) {
                        <aside class="detail-panel" aria-label="Adsorption record details">
                            <div class="detail-header"><div><p class="eyebrow">{{ detail.source }} · {{ detail.external_id }}</p><h3>{{ detail.material }} / {{ detail.adsorbates.join(', ') }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="adsorptionDetail.set(null)">×</button></div>
                            <div class="detail-grid"><div><span>Temperature</span><strong>{{ formatNumber(detail.temperature_k) }} K</strong></div><div><span>Pressure basis</span><strong>{{ detail.pressure_basis }}</strong></div><div><span>Measurements</span><strong>{{ detail.measurements.length }}</strong></div><div><span>Retrieved</span><strong>{{ formatDate(detail.retrieved_at) }}</strong></div></div>
                            <div class="plot-panel">
                                <div class="plot-title"><strong>Isotherm</strong><span>Canonical pressure and uptake</span></div>
                                @if (detail.measurements.length > 1) {
                                    <svg viewBox="0 0 720 300" role="img" aria-label="Pressure versus uptake isotherm plot">
                                        <line x1="62" y1="24" x2="62" y2="250" class="plot-axis"/><line x1="62" y1="250" x2="700" y2="250" class="plot-axis"/>
                                        @for (series of plotSeries(); track series.adsorbate) {
                                            <polyline [attr.points]="series.points" class="plot-line" />
                                        }
                                        <text x="360" y="286" text-anchor="middle">Pressure (Pa)</text><text x="18" y="150" text-anchor="middle" transform="rotate(-90 18 150)">Uptake (mol/kg)</text>
                                    </svg>
                                } @else { <p class="empty-inline">At least two measurements are required for a plot.</p> }
                            </div>
                            <div class="detail-section"><h4>Provenance</h4>@for (identifier of detail.external_identifiers; track identifier.source + identifier.external_id) {<div class="provenance-row"><span>{{ identifier.source }}</span><code>{{ identifier.external_id }}</code>@if (identifier.source_url) {<a [href]="identifier.source_url" target="_blank" rel="noopener noreferrer">Open</a>}</div>}</div>
                        </aside>
                    }
                }
                @case ('materials') {
                    <section class="public-section">
                        <div class="section-heading"><div><p class="eyebrow">Canonical adsorbents</p><h3>Materials</h3><p>Material identities remain separate from source records so multiple providers can reference one canonical material when identity is trustworthy.</p></div></div>
                        <div class="filter-bar compact">
                            <label>Search<input [(ngModel)]="materialQuery" placeholder="Material name" /></label><label>Formula<input [(ngModel)]="materialFormula" placeholder="Formula" /></label><label>Structure<select [(ngModel)]="materialStructure"><option value="">Any</option><option value="yes">Available</option><option value="no">Not available</option></select></label><div class="filter-actions"><button class="button primary" type="button" (click)="applyMaterialFilters()">Apply</button><button class="button secondary" type="button" (click)="resetMaterialFilters()">Reset</button></div>
                        </div>
                        <div class="table-frame"><table><thead><tr><th>Material</th><th>Formula</th><th>Molar mass</th><th>Sources</th><th>Structures</th></tr></thead><tbody>@for (item of materials()?.items ?? []; track item.id) {<tr><td class="truncate-cell" [title]="item.name">{{ item.name }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.molar_mass_g_mol === null ? '—' : formatNumber(item.molar_mass_g_mol) + ' g/mol' }}</td><td>{{ sourceKeys(item.external_identifiers) }}</td><td>{{ item.structure_count }}</td></tr>} @empty {<tr><td colspan="5" class="empty-table">No materials match the current filters.</td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(materials()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(materials()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                }
                @case ('chemicals') {
                    <section class="public-section">
                        <div class="section-heading"><div><p class="eyebrow">Chemical identity</p><h3>Chemicals & adsorbates</h3><p>Resolve PubChem records by CID, InChIKey, or name. Exact identifiers are preferred over display-name matching.</p></div></div>
                        <div class="provider-query"><div><label for="pubchem-query">PubChem lookup</label><span>Retrieve and cache a normalized chemical record.</span></div><input id="pubchem-query" [(ngModel)]="pubchemQuery" placeholder="CID, InChIKey, or compound name" (keyup.enter)="resolveChemical()"/><button class="button primary" type="button" [disabled]="providerBusy() || !pubchemQuery.trim()" (click)="resolveChemical()">Resolve</button></div>
                        <div class="filter-bar compact"><label>Local search<input [(ngModel)]="chemicalQuery" placeholder="Name or synonym" /></label><label>Formula<input [(ngModel)]="chemicalFormula" placeholder="Formula" /></label><label>Min mass<input [(ngModel)]="chemicalMassMin" type="number" min="0" /></label><label>Max mass<input [(ngModel)]="chemicalMassMax" type="number" min="0" /></label><div class="filter-actions"><button class="button primary" type="button" (click)="applyChemicalFilters()">Apply</button><button class="button secondary" type="button" (click)="resetChemicalFilters()">Reset</button></div></div>
                        <div class="table-frame"><table><thead><tr><th>Name</th><th>Formula</th><th>Molecular weight</th><th>InChIKey</th><th>PubChem CID</th><th>Sources</th><th></th></tr></thead><tbody>@for (item of chemicals()?.items ?? []; track item.id) {<tr><td class="truncate-cell" [title]="item.name">{{ item.name }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.molecular_weight === null ? '—' : formatNumber(item.molecular_weight) + ' g/mol' }}</td><td class="mono truncate-cell" [title]="item.inchi_key || ''">{{ item.inchi_key || '—' }}</td><td>{{ item.pubchem_cid || '—' }}</td><td>{{ sourceKeys(item.external_identifiers) }}</td><td><button class="row-action" type="button" (click)="openChemical(item.id)">Inspect</button></td></tr>} @empty {<tr><td colspan="7" class="empty-table">No chemicals match the current filters.</td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(chemicals()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(chemicals()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (chemicalDetail(); as detail) {
                        <aside class="detail-panel"><div class="detail-header"><div><p class="eyebrow">Chemical record</p><h3>{{ detail.preferred_name || detail.name }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="chemicalDetail.set(null)">×</button></div>
                            <div class="chemical-detail-layout">@if (detail.structure_2d_url) {<div class="molecule-preview"><img [src]="detail.structure_2d_url" [alt]="'2D molecular structure for ' + detail.name"/></div>}<div class="detail-grid"><div><span>Formula</span><strong>{{ detail.formula || '—' }}</strong></div><div><span>Molecular weight</span><strong>{{ detail.molecular_weight === null ? '—' : formatNumber(detail.molecular_weight) + ' g/mol' }}</strong></div><div><span>PubChem CID</span><strong>{{ detail.pubchem_cid || '—' }}</strong></div><div><span>3D conformer</span><strong>{{ detail.conformer_3d_url ? 'Available' : 'Not reported' }}</strong></div></div></div>
                            <div class="detail-section"><h4>Stable identifiers</h4><dl class="identifier-list"><dt>InChIKey</dt><dd>{{ detail.inchi_key || '—' }}</dd><dt>InChI</dt><dd>{{ detail.inchi || '—' }}</dd><dt>SMILES</dt><dd>{{ detail.smiles || '—' }}</dd><dt>Connectivity SMILES</dt><dd>{{ detail.connectivity_smiles || '—' }}</dd></dl></div>
                            <div class="detail-section"><h4>Physicochemical properties</h4><div class="property-grid">@for (property of detail.properties; track property.key) {<div><span>{{ property.key }}</span><strong>{{ property.value_number ?? property.value_text }}{{ property.unit ? ' ' + property.unit : '' }}</strong></div>} @empty {<p class="empty-inline">No additional normalized properties are cached.</p>}</div></div>
                            <div class="detail-section"><h4>Synonyms</h4><p class="wrap-list">{{ detail.synonyms.slice(0, 20).join(' · ') || 'No synonyms cached.' }}</p></div>
                        </aside>
                    }
                }
                @case ('structures') {
                    <section class="public-section">
                        <div class="section-heading"><div><p class="eyebrow">Crystallographic records</p><h3>Structures</h3><p>Search COD with a bounded query, inspect crystallographic metadata, then explicitly link imported structures to a canonical material when appropriate.</p></div></div>
                        <div class="provider-query structure-query"><div><label for="cod-query">COD search</label><span>Use a COD ID, formula, or text query. Broad result sets are rejected server-side.</span></div><input id="cod-query" [(ngModel)]="codQuery" placeholder="COD ID, formula, or text" (keyup.enter)="searchStructures()"/><select [(ngModel)]="codQueryType"><option value="text">Text</option><option value="formula">Formula</option><option value="id">COD ID</option></select><button class="button primary" type="button" [disabled]="providerBusy() || !codQuery.trim()" (click)="searchStructures()">Search</button></div>
                        @if (codResults().length) {<div class="provider-results"><div class="section-heading small"><div><h4>COD results</h4><p>Importing stores the original CIF plus normalized unit-cell and atom-site coordinates.</p></div></div><div class="table-frame"><table><thead><tr><th>COD ID</th><th>Name</th><th>Formula</th><th>Space group</th><th>Cell volume</th><th>Coordinates</th><th></th></tr></thead><tbody>@for (item of codResults(); track item.cod_id) {<tr><td>{{ item.cod_id }}</td><td class="truncate-cell" [title]="item.name || ''">{{ item.name || '—' }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.space_group || '—' }}</td><td>{{ item.cell_volume_angstrom3 === null ? '—' : formatNumber(item.cell_volume_angstrom3) + ' Å³' }}</td><td>{{ item.has_coordinates ? 'Yes' : 'Unknown' }}</td><td><button class="row-action" type="button" (click)="importStructure(item)">Import</button></td></tr>}</tbody></table></div></div>} @else if (codSearchCompleted()) {<p class="empty-inline provider-empty-state" role="status">No COD records matched that query. Try a COD ID, formula, or narrower text query.</p>}
                        <div class="filter-bar compact"><label>Local structures<input [(ngModel)]="structureQuery" placeholder="Name or formula" /></label><label>Link status<select [(ngModel)]="structureLink"><option value="">Any</option><option value="linked">Linked to material</option><option value="unlinked">Unlinked</option></select></label><div class="filter-actions"><button class="button primary" type="button" (click)="applyStructureFilters()">Apply</button><button class="button secondary" type="button" (click)="resetStructureFilters()">Reset</button></div></div>
                        <div class="table-frame"><table><thead><tr><th>Source</th><th>Source ID</th><th>Material</th><th>Formula</th><th>Space group</th><th>Atoms</th><th>Retrieved</th><th></th></tr></thead><tbody>@for (item of structures()?.items ?? []; track item.id) {<tr><td><span class="source-key">{{ item.source }}</span></td><td>{{ item.external_id }}</td><td class="truncate-cell">{{ item.material_name || 'Unlinked' }}</td><td>{{ item.formula || '—' }}</td><td>{{ item.space_group || '—' }}</td><td>{{ item.atom_count }}</td><td>{{ formatDate(item.retrieved_at) }}</td><td><button class="row-action" type="button" (click)="openStructure(item.id)">Inspect</button></td></tr>} @empty {<tr><td colspan="8" class="empty-table">No imported structures match the current filters.</td></tr>}</tbody></table></div>
                        <div class="pagination-row">{{ pageSummary(structures()?.pagination) }}<div><button class="button secondary" type="button" [disabled]="page() <= 1" (click)="previousPage()">Previous</button><button class="button secondary" type="button" [disabled]="!hasNextPage(structures()?.pagination)" (click)="nextPage()">Next</button></div></div>
                    </section>
                    @if (structureDetail(); as detail) {<aside class="detail-panel"><div class="detail-header"><div><p class="eyebrow">{{ detail.source }} · {{ detail.external_id }}</p><h3>{{ detail.name || detail.formula || 'Structure record' }}</h3></div><button class="detail-close" type="button" aria-label="Close details" (click)="structureDetail.set(null)">×</button></div><div class="detail-grid"><div><span>Material</span><strong>{{ detail.material_name || 'Not linked' }}</strong></div><div><span>Space group</span><strong>{{ detail.space_group || '—' }}</strong></div><div><span>Unit-cell volume</span><strong>{{ detail.cell_volume_angstrom3 === null ? '—' : formatNumber(detail.cell_volume_angstrom3) + ' Å³' }}</strong></div><div><span>Normalized atoms</span><strong>{{ detail.atom_count }}</strong></div></div><div class="detail-section"><h4>Unit cell</h4><p>a {{ valueOrDash(detail.cell_a_angstrom) }} Å · b {{ valueOrDash(detail.cell_b_angstrom) }} Å · c {{ valueOrDash(detail.cell_c_angstrom) }} Å</p><p>α {{ valueOrDash(detail.cell_alpha_deg) }}° · β {{ valueOrDash(detail.cell_beta_deg) }}° · γ {{ valueOrDash(detail.cell_gamma_deg) }}°</p></div><div class="detail-section"><h4>Atomic coordinates</h4><div class="atom-list">@for (atom of detail.atoms.slice(0, 50); track atom.sequence_index) {<code>{{ atom.label }} {{ atom.element }} ({{ formatNumber(atom.fractional_x) }}, {{ formatNumber(atom.fractional_y) }}, {{ formatNumber(atom.fractional_z) }})</code>} @empty {<p class="empty-inline">No fractional atom coordinates were normalized from this CIF.</p>}</div>@if (detail.atoms.length > 50) {<p class="detail-note">Showing the first 50 of {{ detail.atoms.length }} atom sites.</p>}</div>@if (detail.source_url) {<a class="button secondary external-button" [href]="detail.source_url" target="_blank" rel="noopener noreferrer">Open source record</a>}</aside>}
                }
                @case ('sources') {
                    <section class="public-section">
                        <div class="section-heading"><div><p class="eyebrow">Provider registry</p><h3>Sources & acquisition</h3><p>Provider availability is independent. An outage in one public service does not make locally cached records or other providers unavailable.</p></div><button class="button secondary" type="button" (click)="refreshSources()">Refresh status</button></div>
                        <div class="source-table">
                            @for (source of sources(); track source.key) {
                                <article class="source-row"><div class="source-health"><span class="health-dot" [class.available]="source.status === 'available'" [class.unavailable]="source.status === 'unavailable'"></span><div><strong>{{ source.name }}</strong><code>{{ source.key }}</code></div></div><p>{{ source.description }}</p><div class="capability-list">@for (capability of source.capabilities; track capability) {<span>{{ capability }}</span>}</div><div class="source-meta"><span>{{ source.record_count }} cached provenance records</span><span>Checked {{ formatDate(source.last_checked_at) }}</span>@if (source.license_name) {<span>{{ source.license_name }}</span>}</div><div class="source-actions"><a [href]="source.homepage_url" target="_blank" rel="noopener noreferrer">Source</a>@if (source.license_url) {<a [href]="source.license_url" target="_blank" rel="noopener noreferrer">License</a>}</div>@if (source.status_detail) {<p class="source-detail">{{ source.status_detail }}</p>}</article>
                            }
                        </div>
                    </section>
                    <section class="public-section acquisition-section"><div class="section-heading"><div><p class="eyebrow">NIST acquisition</p><h3>Download NIST collections</h3><p>The existing NIST jobs remain the ingestion mechanism for experiments, guest species, and host materials. Retrieved records are normalized into the shared provenance model.</p></div></div><adsmod-nist-collection-rows [categories]="nistCategories" (statusUpdate)="appendStatus($event)" /></section>
                    <section class="public-section activity-section"><div class="section-heading"><div><p class="eyebrow">Activity</p><h3>Retrieval status</h3></div><button class="button secondary" type="button" (click)="statusMessages.set([])">Clear</button></div>@if (statusMessages().length) {<pre class="activity-log">{{ statusMessages().join('\n\n') }}</pre>} @else {<p class="empty-inline">NIST acquisition updates will appear here.</p>}</section>
                }
            }
        </div>
    `,
    styleUrl: './public-data-workspace.component.css',
})
export class PublicDataWorkspaceComponent {
    private readonly route = inject(ActivatedRoute);
    private readonly router = inject(Router);
    private readonly destroyRef = inject(DestroyRef);

    protected readonly views = VIEWS;
    protected readonly view = signal<PublicDataView>('overview');
    protected readonly loading = signal(false);
    protected readonly providerBusy = signal(false);
    protected readonly error = signal<string | null>(null);
    protected readonly sources = signal<PublicSourceSummary[]>([]);
    protected readonly adsorption = signal<AdsorptionPageResponse | null>(null);
    protected readonly adsorptionDetail = signal<AdsorptionDetailResponse | null>(null);
    protected readonly materials = signal<MaterialPageResponse | null>(null);
    protected readonly chemicals = signal<ChemicalPageResponse | null>(null);
    protected readonly chemicalDetail = signal<ChemicalRecordView | null>(null);
    protected readonly structures = signal<StructurePageResponse | null>(null);
    protected readonly structureDetail = signal<StructureRecordView | null>(null);
    protected readonly codResults = signal<CODSearchResult[]>([]);
    protected readonly codSearchCompleted = signal(false);
    protected readonly statusMessages = signal<string[]>([]);
    protected readonly page = signal(1);
    protected readonly pageSize = 25;
    protected readonly nistCategories = ['experiments', 'guest', 'host'] as const;

    protected readonly overviewAdsorptionTotal = signal(0);
    protected readonly overviewMaterialTotal = signal(0);
    protected readonly overviewChemicalTotal = signal(0);
    protected readonly overviewStructureTotal = signal(0);

    protected adsorptionSource = '';
    protected adsorptionMaterial = '';
    protected adsorptionAdsorbate = '';
    protected temperatureMin: number | null = null;
    protected temperatureMax: number | null = null;
    protected materialQuery = '';
    protected materialFormula = '';
    protected materialStructure = '';
    protected chemicalQuery = '';
    protected chemicalFormula = '';
    protected chemicalMassMin: number | null = null;
    protected chemicalMassMax: number | null = null;
    protected pubchemQuery = '';
    protected codQuery = '';
    protected codQueryType: 'text' | 'formula' | 'id' = 'text';
    protected structureQuery = '';
    protected structureLink = '';

    protected readonly plotSeries = computed(() => {
        const measurements = this.adsorptionDetail()?.measurements ?? [];
        if (measurements.length < 2) return [];
        const pressures = measurements.map((item) => item.pressure_pa);
        const uptakes = measurements.map((item) => item.uptake_mol_kg);
        const minX = Math.min(...pressures);
        const maxX = Math.max(...pressures);
        const minY = Math.min(...uptakes);
        const maxY = Math.max(...uptakes);
        const spanX = Math.max(maxX - minX, Number.EPSILON);
        const spanY = Math.max(maxY - minY, Number.EPSILON);
        const grouped = new Map<string, typeof measurements>();
        for (const item of measurements) {
            const series = grouped.get(item.adsorbate) ?? [];
            series.push(item);
            grouped.set(item.adsorbate, series);
        }
        return [...grouped.entries()].map(([adsorbate, series]) => ({
            adsorbate,
            points: series
                .sort((left, right) => left.pressure_pa - right.pressure_pa)
                .map((item) => {
                    const x = 62 + ((item.pressure_pa - minX) / spanX) * 638;
                    const y = 250 - ((item.uptake_mol_kg - minY) / spanY) * 226;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(' '),
        }));
    });

    constructor() {
        this.route.paramMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
            const rawView = params.get('view') ?? 'overview';
            const nextView = VIEWS.includes(rawView as PublicDataView) ? (rawView as PublicDataView) : 'overview';
            if (nextView !== rawView) {
                void this.router.navigate(['/public-data', 'overview'], { replaceUrl: true });
                return;
            }
            this.view.set(nextView);
            this.page.set(1);
            this.closeDetails();
            void this.loadCurrent();
        });
    }

    protected viewLabel(view: PublicDataView): string {
        return ({ overview: 'Overview', adsorption: 'Adsorption Data', materials: 'Materials', chemicals: 'Chemicals', structures: 'Structures', sources: 'Sources' } satisfies Record<PublicDataView, string>)[view];
    }

    protected async reload(): Promise<void> { await this.loadCurrent(); }

    private async ensureSources(checkHealth = false): Promise<void> {
        if (this.sources().length && !checkHealth) return;
        const result = await fetchPublicSources(checkHealth);
        if (result.data) this.sources.set(result.data.sources);
        else if (checkHealth) this.error.set(result.error);
    }

    private async loadCurrent(): Promise<void> {
        this.loading.set(true);
        this.error.set(null);
        try {
            await this.ensureSources(this.view() === 'overview' || this.view() === 'sources');
            switch (this.view()) {
                case 'overview': await this.loadOverview(); break;
                case 'adsorption': await this.loadAdsorption(); break;
                case 'materials': await this.loadMaterials(); break;
                case 'chemicals': await this.loadChemicals(); break;
                case 'structures': await this.loadStructures(); break;
                case 'sources': break;
            }
        } finally {
            this.loading.set(false);
        }
    }

    private async loadOverview(): Promise<void> {
        const [adsorption, materials, chemicals, structures] = await Promise.all([
            fetchPublicAdsorption({ page: 1, page_size: 1 }),
            fetchPublicMaterials({ page: 1, page_size: 1 }),
            fetchPublicChemicals({ page: 1, page_size: 1 }),
            fetchPublicStructures({ page: 1, page_size: 1 }),
        ]);
        this.overviewAdsorptionTotal.set(adsorption.data?.pagination.total ?? 0);
        this.overviewMaterialTotal.set(materials.data?.pagination.total ?? 0);
        this.overviewChemicalTotal.set(chemicals.data?.pagination.total ?? 0);
        this.overviewStructureTotal.set(structures.data?.pagination.total ?? 0);
        this.error.set(adsorption.error || materials.error || chemicals.error || structures.error);
    }

    private async loadAdsorption(): Promise<void> {
        const result = await fetchPublicAdsorption({ page: this.page(), page_size: this.pageSize, source: this.adsorptionSource, material: this.adsorptionMaterial, adsorbate: this.adsorptionAdsorbate, temperature_min_k: this.temperatureMin, temperature_max_k: this.temperatureMax });
        this.adsorption.set(result.data);
        this.error.set(result.error);
    }

    private async loadMaterials(): Promise<void> {
        const hasStructure = this.materialStructure === 'yes' ? true : this.materialStructure === 'no' ? false : null;
        const result = await fetchPublicMaterials({ page: this.page(), page_size: this.pageSize, q: this.materialQuery, formula: this.materialFormula, has_structure: hasStructure });
        this.materials.set(result.data);
        this.error.set(result.error);
    }

    private async loadChemicals(): Promise<void> {
        const result = await fetchPublicChemicals({ page: this.page(), page_size: this.pageSize, q: this.chemicalQuery, formula: this.chemicalFormula, molecular_weight_min: this.chemicalMassMin, molecular_weight_max: this.chemicalMassMax });
        this.chemicals.set(result.data);
        this.error.set(result.error);
    }

    private async loadStructures(): Promise<void> {
        const linked = this.structureLink === 'linked' ? true : this.structureLink === 'unlinked' ? false : null;
        const result = await fetchPublicStructures({ page: this.page(), page_size: this.pageSize, q: this.structureQuery, linked_only: linked });
        this.structures.set(result.data);
        this.error.set(result.error);
    }

    protected async applyAdsorptionFilters(): Promise<void> { this.page.set(1); await this.loadAdsorption(); }
    protected async resetAdsorptionFilters(): Promise<void> { this.adsorptionSource = ''; this.adsorptionMaterial = ''; this.adsorptionAdsorbate = ''; this.temperatureMin = null; this.temperatureMax = null; await this.applyAdsorptionFilters(); }
    protected async applyMaterialFilters(): Promise<void> { this.page.set(1); await this.loadMaterials(); }
    protected async resetMaterialFilters(): Promise<void> { this.materialQuery = ''; this.materialFormula = ''; this.materialStructure = ''; await this.applyMaterialFilters(); }
    protected async applyChemicalFilters(): Promise<void> { this.page.set(1); await this.loadChemicals(); }
    protected async resetChemicalFilters(): Promise<void> { this.chemicalQuery = ''; this.chemicalFormula = ''; this.chemicalMassMin = null; this.chemicalMassMax = null; await this.applyChemicalFilters(); }
    protected async applyStructureFilters(): Promise<void> { this.page.set(1); await this.loadStructures(); }
    protected async resetStructureFilters(): Promise<void> { this.structureQuery = ''; this.structureLink = ''; await this.applyStructureFilters(); }

    protected async previousPage(): Promise<void> { if (this.page() > 1) { this.page.update((value) => value - 1); await this.loadCurrent(); } }
    protected async nextPage(): Promise<void> { this.page.update((value) => value + 1); await this.loadCurrent(); }

    protected async openAdsorption(id: number): Promise<void> { const result = await fetchPublicAdsorptionDetail(id); this.adsorptionDetail.set(result.data); this.error.set(result.error); }
    protected async openChemical(id: number): Promise<void> { const result = await fetchPublicChemicalDetail(id); this.chemicalDetail.set(result.data); this.error.set(result.error); }
    protected async openStructure(id: number): Promise<void> { const result = await fetchPublicStructureDetail(id); this.structureDetail.set(result.data); this.error.set(result.error); }

    protected async resolveChemical(): Promise<void> {
        const query = this.pubchemQuery.trim();
        if (!query) return;
        this.providerBusy.set(true); this.error.set(null);
        try {
            const result = await resolvePubChem(query);
            this.chemicalDetail.set(result.data);
            this.error.set(result.error);
            if (result.data) { this.pubchemQuery = ''; await this.loadChemicals(); }
        } finally { this.providerBusy.set(false); }
    }

    protected async searchStructures(): Promise<void> {
        const query = this.codQuery.trim();
        if (!query) return;
        this.providerBusy.set(true); this.error.set(null); this.codSearchCompleted.set(false); this.codResults.set([]);
        try {
            const params = this.codQueryType === 'id' ? { cod_id: query } : this.codQueryType === 'formula' ? { formula: query } : { q: query };
            const result = await searchCOD(params);
            this.codResults.set(result.data?.items ?? []);
            this.codSearchCompleted.set(true);
            this.error.set(result.error);
        } finally { this.providerBusy.set(false); }
    }

    protected async importStructure(item: CODSearchResult): Promise<void> {
        this.providerBusy.set(true); this.error.set(null);
        try {
            const result = await importCOD(item.cod_id);
            this.structureDetail.set(result.data);
            this.error.set(result.error);
            if (result.data) await this.loadStructures();
        } finally { this.providerBusy.set(false); }
    }

    protected async refreshSources(): Promise<void> { this.loading.set(true); this.error.set(null); try { await this.ensureSources(true); } finally { this.loading.set(false); } }
    protected appendStatus(message: string): void { this.statusMessages.update((entries) => [...entries, message]); void this.refreshSources(); }

    private closeDetails(): void { this.adsorptionDetail.set(null); this.chemicalDetail.set(null); this.structureDetail.set(null); }

    protected sourceKeys(identifiers: { source: string }[]): string { return [...new Set(identifiers.map((item) => item.source))].join(', ') || '—'; }
    protected formatNumber(value: number): string { return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 }).format(value); }
    protected valueOrDash(value: number | null): string { return value === null ? '—' : this.formatNumber(value); }
    protected formatDate(value: string | null): string { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date); }
    protected range(min: number | null, max: number | null, unit: string): string { return min === null || max === null ? '—' : `${this.formatNumber(min)}–${this.formatNumber(max)} ${unit}`; }
    protected pageSummary(pagination: { page: number; page_size: number; total: number } | undefined): string { if (!pagination || pagination.total === 0) return '0 records'; const first = (pagination.page - 1) * pagination.page_size + 1; const last = Math.min(pagination.total, pagination.page * pagination.page_size); return `${first}–${last} of ${pagination.total}`; }
    protected hasNextPage(pagination: { page: number; page_size: number; total: number } | undefined): boolean { return !!pagination && pagination.page * pagination.page_size < pagination.total; }
}
