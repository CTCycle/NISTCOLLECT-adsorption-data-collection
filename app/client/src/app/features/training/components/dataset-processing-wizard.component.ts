import { Component, OnInit, computed, input, output, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import type { DatasetBuildConfig, DatasetSelection } from '../../../models/dataset-build.model';
import type { DatasetSourceInfo, NumericConstraint } from '../../../models/training.model';
import { NumberInputComponent } from '../../../shared/components/number-input/number-input.component';
import { WizardProgressIndicatorComponent } from './wizard-progress-indicator.component';

const buildDatasetKey = (dataset: DatasetSourceInfo): string => `${dataset.source}:${dataset.dataset_name}`;

@Component({
    selector: 'adsmod-dataset-processing-wizard',
    standalone: true,
    imports: [ReactiveFormsModule, NumberInputComponent, WizardProgressIndicatorComponent],
    template: `
        <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="dataset-processing-wizard-title">
            <div class="wizard-modal">
                <div class="wizard-header">
                    <h4 id="dataset-processing-wizard-title">Dataset Processing Wizard</h4>
                    <p>Configure processing settings for your selected datasets.</p>
                    <adsmod-wizard-progress-indicator [currentPage]="currentPage()" [totalPages]="2" />
                </div>

                <div class="wizard-body">
                    @if (currentPage() === 0) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header">
                                    <span class="wizard-card-icon">⚙️</span>
                                    <span>Processing Settings</span>
                                </div>
                                <p class="wizard-card-description">
                                    Configure the parameters for dataset preprocessing. These settings control
                                    how raw adsorption data is filtered, sampled, and split for training.
                                </p>
                                <div class="wizard-card-body">
                                    <div class="wizard-settings-grid">
                                        <adsmod-number-input label="Sample Size" [value]="sampleSize()" [min]="minimum('dataset_sample_size')" [max]="maximum('dataset_sample_size')" [step]="0.01" [precision]="2" (valueChange)="sampleSizeControl.setValue($event)" />
                                        <adsmod-number-input label="Validation %" [value]="validationSize()" [min]="minimum('dataset_validation_size')" [max]="maximum('dataset_validation_size')" [step]="0.05" [precision]="2" (valueChange)="validationSizeControl.setValue($event)" />
                                        <adsmod-number-input label="SMILE Length" [value]="smileSequenceSize()" [min]="minimum('dataset_smile_sequence_size')" [max]="maximum('dataset_smile_sequence_size')" [step]="5" [precision]="0" (valueChange)="smileSequenceSizeControl.setValue($event)" />
                                        <adsmod-number-input label="Min Measurements" [value]="minMeasurements()" [min]="minimum('dataset_min_measurements')" [max]="maximum('dataset_min_measurements')" [step]="1" [precision]="0" (valueChange)="minMeasurementsControl.setValue($event)" />
                                        <adsmod-number-input label="Max Measurements" [value]="maxMeasurements()" [min]="minimum('dataset_max_measurements')" [max]="maximum('dataset_max_measurements')" [step]="5" [precision]="0" (valueChange)="maxMeasurementsControl.setValue($event)" />
                                        <adsmod-number-input label="Max Pressure (kPa)" [value]="maxPressure()" [min]="minimum('dataset_max_pressure')" [max]="maximum('dataset_max_pressure')" [step]="100" [precision]="0" (valueChange)="maxPressureControl.setValue($event)" />
                                        <adsmod-number-input label="Max Uptake (mol/g)" [value]="maxUptake()" [min]="minimum('dataset_max_uptake')" [max]="maximum('dataset_max_uptake')" [step]="1" [precision]="1" (valueChange)="maxUptakeControl.setValue($event)" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    } @else {
                        <div class="wizard-page">
                            <div class="wizard-card wizard-name-card">
                                <div class="wizard-card-header">
                                    <span class="wizard-card-icon">🏷️</span>
                                    <span>Dataset Name</span>
                                </div>
                                <div class="wizard-card-body">
                                    <div class="wizard-name-field">
                                        <label class="field-label" for="processed-dataset-name">Custom Name</label>
                                        <input
                                            id="processed-dataset-name"
                                            type="text"
                                            [formControl]="datasetNameControl"
                                            placeholder="e.g. my_dataset_v1"
                                            class="number-input-field"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div class="wizard-summary">
                                <div class="wizard-summary-section">
                                    <h5>Selected Datasets</h5>
                                    <ul>
                                        @for (dataset of selectedDatasets(); track datasetKey(dataset)) {
                                            <li>
                                                <strong>{{ dataset.display_name }}</strong>
                                                <span class="wizard-summary-meta">{{ dataset.source }} • {{ dataset.row_count }} rows</span>
                                            </li>
                                        }
                                    </ul>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Processing Settings</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Sample size</span><strong>{{ sampleSize() }}</strong>
                                        <span>Validation split</span><strong>{{ validationSize() }}</strong>
                                        <span>SMILE length</span><strong>{{ smileSequenceSize() }}</strong>
                                        <span>Min measurements</span><strong>{{ minMeasurements() }}</strong>
                                        <span>Max measurements</span><strong>{{ maxMeasurements() }}</strong>
                                        <span>Max pressure (kPa)</span><strong>{{ maxPressure() }}</strong>
                                        <span>Max uptake (mol/g)</span><strong>{{ maxUptake() }}</strong>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                </div>

                <div class="wizard-footer">
                    <button class="secondary" type="button" (click)="closed.emit()">Cancel</button>
                    @if (currentPage() === 0) {
                        <button class="primary" type="button" (click)="currentPage.set(1)">Next →</button>
                    } @else {
                        <button class="secondary" type="button" (click)="currentPage.set(0)">← Previous</button>
                        <button class="primary" type="button" [disabled]="form.invalid || selectedDatasets().length === 0" (click)="submit()">✓ Build Dataset</button>
                    }
                </div>
            </div>
        </div>
    `,
})
export class DatasetProcessingWizardComponent implements OnInit {
    readonly selectedDatasets = input.required<DatasetSourceInfo[]>();
    readonly initialConfig = input.required<Partial<DatasetBuildConfig>>();
    readonly numericConstraints = input<Record<string, NumericConstraint>>({});
    readonly closed = output<void>();
    readonly buildStarted = output<DatasetBuildConfig>();
    protected readonly currentPage = signal(0);

    protected readonly sampleSizeControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly validationSizeControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly minMeasurementsControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly maxMeasurementsControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly smileSequenceSizeControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly maxPressureControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly maxUptakeControl = new FormControl<number | null>(null, { validators: [Validators.required] });
    protected readonly datasetNameControl = new FormControl(this.defaultDatasetName(), { nonNullable: true });
    protected readonly form = new FormGroup({
        sample_size: this.sampleSizeControl,
        validation_size: this.validationSizeControl,
        min_measurements: this.minMeasurementsControl,
        max_measurements: this.maxMeasurementsControl,
        smile_sequence_size: this.smileSequenceSizeControl,
        max_pressure: this.maxPressureControl,
        max_uptake: this.maxUptakeControl,
        dataset_label: this.datasetNameControl,
    });

    protected readonly sampleSize = computed(() => this.sampleSizeControl.value);
    protected readonly validationSize = computed(() => this.validationSizeControl.value);
    protected readonly minMeasurements = computed(() => this.minMeasurementsControl.value);
    protected readonly maxMeasurements = computed(() => this.maxMeasurementsControl.value);
    protected readonly smileSequenceSize = computed(() => this.smileSequenceSizeControl.value);
    protected readonly maxPressure = computed(() => this.maxPressureControl.value);
    protected readonly maxUptake = computed(() => this.maxUptakeControl.value);

    protected minimum(field: string): number | undefined {
        return this.numericConstraints()[field]?.minimum;
    }

    protected maximum(field: string): number | undefined {
        return this.numericConstraints()[field]?.maximum;
    }

    protected datasetKey(dataset: DatasetSourceInfo): string {
        return buildDatasetKey(dataset);
    }

    protected submit(): void {
        if (this.form.invalid || Object.values(this.form.getRawValue()).some((value) => value === null || value === undefined)) {
            this.form.markAllAsTouched();
            return;
        }
        const datasets: DatasetSelection[] = this.selectedDatasets().map((dataset) => ({
            source: dataset.source,
            dataset_name: dataset.dataset_name,
            dataset_id: dataset.dataset_id,
        }));
        const formValue = this.form.getRawValue() as Omit<DatasetBuildConfig, 'datasets'>;
        this.closed.emit();
        this.buildStarted.emit({
            ...formValue,
            datasets,
            dataset_label: formValue.dataset_label || undefined,
        });
    }

    private defaultDatasetName(): string {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        return `dataset_${timestamp}`;
    }

    ngOnInit(): void {
        this.form.patchValue(this.initialConfig());
    }
}
