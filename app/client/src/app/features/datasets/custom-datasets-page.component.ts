import { CommonModule } from '@angular/common';
import { Component, inject, signal } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import { DatasetManagementComponent } from '../source/dataset-management.component';
import { DatasetImportWizardComponent } from '../source/dataset-import-wizard.component';
import { fetchDatasetConfiguration } from '../../services/dataset.service';

const DEFAULT_FILE_ACCEPT = '.csv,.xls,.xlsx';

@Component({
    selector: 'adsmod-custom-datasets-page',
    standalone: true,
    imports: [CommonModule, DatasetManagementComponent, DatasetImportWizardComponent],
    template: `
        <div class="data-page custom-datasets-page">
            <div class="page-intro-card console-card">
                <p class="eyebrow">Workspace data</p>
                <h2>Custom Datasets</h2>
                <p>Upload, inspect, and maintain datasets provided by your team. Public source collections are managed on their dedicated pages.</p>
            </div>
            <input #sourceFileInput class="source-file-input" type="file" [accept]="acceptedFileTypes()" (change)="fileChanged($event)" />
            @if (store.managementStatus()) {
                <p class="dataset-status error" role="alert">{{ store.managementStatus() }}</p>
            }
            <adsmod-dataset-management
                [datasets]="store.customDatasets()"
                [selected]="store.selectedDatasetId()"
                (addRequested)="sourceFileInput.click()"
                (opened)="store.selectDataset($event)"
                (deleted)="store.deleteDataset($event)"
                (renamed)="store.renameDataset($event.id, $event.newName)"
                (metadataSaved)="store.saveMetadata($event.id, $event.metadata)"
            />
            @if (pendingFile(); as file) {
                <adsmod-dataset-import-wizard [file]="file" (closed)="pendingFile.set(null)" (saved)="wizardSaved()" />
            }
        </div>
    `,
})
export class CustomDatasetsPageComponent {
    protected readonly store = inject(CoreWorkspaceStore);
    protected readonly pendingFile = signal<File | null>(null);
    protected readonly acceptedFileTypes = signal(DEFAULT_FILE_ACCEPT);

    constructor() {
        void this.loadDatasetConfiguration();
    }

    private async loadDatasetConfiguration(): Promise<void> {
        const result = await fetchDatasetConfiguration();
        if (result.data?.allowed_extensions?.length) {
            this.acceptedFileTypes.set(result.data.allowed_extensions.join(','));
        }
    }

    protected fileChanged(event: Event): void {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
            this.pendingFile.set(file);
        }
    }

    protected wizardSaved(): void {
        this.pendingFile.set(null);
        void this.store.refreshDatasets();
    }
}
