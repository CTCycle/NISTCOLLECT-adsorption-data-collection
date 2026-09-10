import { TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ExperimentSummary } from '../../models/dataset.model';
import type {
    FittingConfiguration,
    FittingResponse,
} from '../../models/fitting.model';
import { CoreWorkspaceStore } from './core-workspace.store';

const fetchMock = vi.fn();

type MockResponse = {
    ok: boolean;
    json: () => Promise<unknown>;
};

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((promiseResolve, promiseReject) => {
        resolve = promiseResolve;
        reject = promiseReject;
    });
    return { promise, resolve, reject };
}

function response(body: unknown, ok = true): MockResponse {
    return { ok, json: async () => body };
}

function experiment(datasetId: number, id: number, name = `experiment-${id}`): ExperimentSummary {
    return {
        id,
        dataset_id: datasetId,
        external_key: `external-${id}`,
        name,
        adsorbent: 'carbon',
        adsorbates: ['CO2'],
        temperature_k: 298.15,
        pressure_basis: 'absolute',
        observation_count: 2,
        fitting_eligible: true,
        ineligibility_reason: null,
    };
}

function fittingResponse(datasetId: number, experimentId: number): FittingResponse {
    return {
        status: 'success',
        run_id: 1,
        dataset_id: datasetId,
        isotherm_id: experimentId,
        dataset_name: `dataset-${datasetId}`,
        experiment_name: `experiment-${experimentId}`,
        adsorbent: 'carbon',
        adsorbate: 'CO2',
        temperature_k: 298.15,
        pressure_basis: 'absolute',
        pressure_unit: 'bar',
        uptake_unit: 'mmol/g',
        observation_count: 2,
        best_model: null,
        results: [],
        summary: 'completed',
    };
}

const fittingConfiguration: FittingConfiguration = {
    status: 'success',
    supported_optimizers: ['trf'],
    default_optimizer: 'trf',
    default_max_evaluations: 100,
    max_evaluations_bounds: { minimum: 1, maximum: 1000 },
    weighting_options: ['unweighted'],
    default_weighting: 'unweighted',
    display_units: {
        pressure: ['bar'],
        uptake: ['mmol/g'],
        default_pressure: 'bar',
        default_uptake: 'mmol/g',
    },
    parameter_defaults: { lower: 0, upper: 1, initial: 0.5 },
};

const datasetResponse = response({
    datasets: [
        {
            id: 1,
            name: 'dataset-1',
            source: 'uploaded',
            created_at: '2026-01-01T00:00:00Z',
            experiment_count: 1,
            observation_count: 2,
            tags: [],
            description: '',
        },
        {
            id: 2,
            name: 'dataset-2',
            source: 'nist',
            created_at: '2026-01-02T00:00:00Z',
            experiment_count: 1,
            observation_count: 2,
            tags: [],
            description: '',
        },
    ],
});

function setupFetchMock(
    experimentResponses: Record<number, MockResponse | ReturnType<typeof deferred<MockResponse>>>,
    fittingStartResponse?: MockResponse | ReturnType<typeof deferred<MockResponse>>,
    fittingStatusResponse?: MockResponse | ReturnType<typeof deferred<MockResponse>>,
): void {
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/datasets')) {
            return Promise.resolve(datasetResponse);
        }
        if (url.endsWith('/system/configuration')) {
            return Promise.resolve(response(fittingConfiguration));
        }
        if (url.includes('/fitting/models')) {
            return Promise.resolve(response({
                status: 'success',
                pressure_unit: 'bar',
                uptake_unit: 'mmol/g',
                models: [],
            }));
        }

        const experimentMatch = url.match(/\/datasets\/(\d+)\/experiments$/);
        if (experimentMatch) {
            const request = experimentResponses[Number(experimentMatch[1])];
            if (!request) {
                throw new Error(`Unhandled experiment request ${url}`);
            }
            return 'promise' in request
                ? request.promise
                : Promise.resolve(request);
        }
        if (url.endsWith('/fitting/run')) {
            if (!fittingStartResponse) {
                throw new Error(`Unhandled fitting start request ${url}`);
            }
            return 'promise' in fittingStartResponse
                ? fittingStartResponse.promise
                : Promise.resolve(fittingStartResponse);
        }
        if (url.includes('/fitting/jobs/')) {
            if (!fittingStatusResponse) {
                throw new Error(`Unhandled fitting status request ${url}`);
            }
            return 'promise' in fittingStatusResponse
                ? fittingStatusResponse.promise
                : Promise.resolve(fittingStatusResponse);
        }
        throw new Error(`Unhandled URL ${url}`);
    });
}

function prepareFittingState(store: CoreWorkspaceStore): void {
    store.fittingConfiguration.set(fittingConfiguration);
    store.optimizationMethod.set('trf');
    store.maxEvaluations.set(100);
    store.weighting.set('unweighted');
    store.modelStates.set({
        langmuir: { enabled: true, config: {} },
    });
    store.selectedDatasetId.set(1);
    store.experiments.set([experiment(1, 101)]);
    store.selectedExperimentId.set(101);
}

async function flushMicrotasks(): Promise<void> {
    await new Promise<void>((resolve) => queueMicrotask(resolve));
    await Promise.resolve();
}

describe('CoreWorkspaceStore', () => {
    beforeEach(() => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    it('keeps all datasets for fitting while exposing uploaded-only custom datasets', async () => {
        setupFetchMock({});

        const store = TestBed.inject(CoreWorkspaceStore);
        await store.refreshDatasets();

        expect(store.datasets().map((dataset) => dataset.source)).toEqual(['uploaded', 'nist']);
        expect(store.customDatasets().map((dataset) => dataset.name)).toEqual(['dataset-1']);
    });

    it('keeps the latest dataset experiments when responses resolve out of order', async () => {
        const datasetA = deferred<MockResponse>();
        const datasetB = deferred<MockResponse>();
        setupFetchMock({ 1: datasetA, 2: datasetB });

        const store = TestBed.inject(CoreWorkspaceStore);
        const selectA = store.selectDataset(1);
        const selectB = store.selectDataset(2);

        datasetB.resolve(response({ experiments: [experiment(2, 202)] }));
        await selectB;
        datasetA.resolve(response({ experiments: [experiment(1, 101)] }));
        await selectA;

        expect(store.selectedDatasetId()).toBe(2);
        expect(store.experiments()).toEqual([experiment(2, 202)]);
        expect(store.selectedExperimentId()).toBe(202);
    });

    it('keeps experiments empty when a dataset request is invalidated by null selection', async () => {
        const datasetA = deferred<MockResponse>();
        setupFetchMock({ 1: datasetA });

        const store = TestBed.inject(CoreWorkspaceStore);
        const selectA = store.selectDataset(1);
        await store.selectDataset(null);

        datasetA.resolve(response({ experiments: [experiment(1, 101)] }));
        await selectA;

        expect(store.selectedDatasetId()).toBeNull();
        expect(store.experiments()).toEqual([]);
        expect(store.selectedExperimentId()).toBeNull();
        expect(store.experimentsLoading()).toBe(false);
    });

    it('ignores a late failure from an older dataset request', async () => {
        const datasetA = deferred<MockResponse>();
        const datasetB = deferred<MockResponse>();
        setupFetchMock({ 1: datasetA, 2: datasetB });

        const store = TestBed.inject(CoreWorkspaceStore);
        const selectA = store.selectDataset(1);
        const selectB = store.selectDataset(2);

        datasetB.resolve(response({ experiments: [experiment(2, 202)] }));
        await selectB;
        datasetA.reject(new Error('dataset A failed'));
        await selectA;

        expect(store.selectedDatasetId()).toBe(2);
        expect(store.experiments()).toEqual([experiment(2, 202)]);
        expect(store.managementStatus()).not.toBe('dataset A failed');
        expect(store.experimentsLoading()).toBe(false);
    });

    it('auto-selects the only experiment returned for the latest dataset', async () => {
        const datasetA = deferred<MockResponse>();
        const datasetB = deferred<MockResponse>();
        setupFetchMock({ 1: datasetA, 2: datasetB });

        const store = TestBed.inject(CoreWorkspaceStore);
        const selectA = store.selectDataset(1);
        const selectB = store.selectDataset(2);

        datasetB.resolve(response({ experiments: [experiment(2, 202)] }));
        await selectB;
        expect(store.selectedExperimentId()).toBe(202);

        datasetA.resolve(response({ experiments: [experiment(1, 101)] }));
        await selectA;
        expect(store.selectedDatasetId()).toBe(2);
        expect(store.selectedExperimentId()).toBe(202);
    });

    it('does not install a fitting result after the workspace context changes', async () => {
        const fittingStart = deferred<MockResponse>();
        const fittingStatus = deferred<MockResponse>();
        setupFetchMock(
            { 2: response({ experiments: [experiment(2, 202)] }) },
            fittingStart,
            fittingStatus,
        );

        const store = TestBed.inject(CoreWorkspaceStore);
        prepareFittingState(store);
        const fitting = store.startFitting();

        fittingStart.resolve(response({ job_id: 'job-a', poll_interval: 1 }));
        await flushMicrotasks();
        await vi.waitFor(() => {
            expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/fitting/jobs/job-a'))).toBe(true);
        });

        await store.selectDataset(2);
        fittingStatus.resolve(response({
            status: 'completed',
            result: fittingResponse(1, 101),
        }));
        await fitting;

        expect(store.selectedDatasetId()).toBe(2);
        expect(store.fittingRunning()).toBe(false);
        expect(store.fittingResult()).toBeNull();
    });

    it('installs a fitting result when its dataset and experiment remain current', async () => {
        const fittingStart = deferred<MockResponse>();
        const fittingStatus = deferred<MockResponse>();
        setupFetchMock({}, fittingStart, fittingStatus);

        const store = TestBed.inject(CoreWorkspaceStore);
        prepareFittingState(store);
        const fitting = store.startFitting();

        fittingStart.resolve(response({ job_id: 'job-a', poll_interval: 1 }));
        await flushMicrotasks();
        await vi.waitFor(() => {
            expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/fitting/jobs/job-a'))).toBe(true);
        });

        const result = fittingResponse(1, 101);
        fittingStatus.resolve(response({ status: 'completed', result }));
        await fitting;

        expect(store.fittingResult()).toEqual(result);
        expect(store.fittingRunning()).toBe(false);
    });
});
