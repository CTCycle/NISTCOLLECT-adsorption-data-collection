import { API_BASE_URL } from '../core/config/api-base-url';
import type {
    DatasetImportResponse,
    DatasetMetadata,
    DatasetSummary,
    ExperimentSummary,
    ImportMapping,
    ImportPreview,
    ImportValidation,
    ObservationPage,
} from '../models/dataset.model';
import {
    extractErrorMessage,
    fetchWithTimeout,
    HTTP_TIMEOUT,
} from './http-timeout.service';

type ApiResult<T> = { data: T | null; error: string | null };

async function request<T>(
    path: string,
    init: RequestInit,
): Promise<ApiResult<T>> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}/datasets${path}`,
            init,
            HTTP_TIMEOUT,
        );
        const data: unknown =
            response.status === 204
                ? null
                : await response.json().catch(() => ({}));
        return response.ok
            ? { data: data as T, error: null }
            : {
                  data: null,
                  error: extractErrorMessage(response, data),
              };
    } catch (error) {
        return {
            data: null,
            error:
                error instanceof Error
                    ? error.message
                    : 'Unable to reach the backend.',
        };
    }
}

function importForm(file: File, mapping?: ImportMapping): FormData {
    const form = new FormData();
    form.append('file', file);
    if (mapping) {
        form.append('mapping', JSON.stringify(mapping));
    }
    return form;
}

export function previewDataset(file: File): Promise<ApiResult<ImportPreview>> {
    return request('/import/preview', {
        method: 'POST',
        body: importForm(file),
    });
}

export function validateDataset(
    file: File,
    mapping: ImportMapping,
): Promise<ApiResult<ImportValidation>> {
    return request('/import/validate', {
        method: 'POST',
        body: importForm(file, mapping),
    });
}

export function commitDataset(
    file: File,
    mapping: ImportMapping,
): Promise<ApiResult<DatasetImportResponse>> {
    return request('/import/commit', {
        method: 'POST',
        body: importForm(file, mapping),
    });
}

export function fetchDatasets(): Promise<
    ApiResult<{ datasets: DatasetSummary[] }>
> {
    return request('', { method: 'GET' });
}

export function fetchDatasetConfiguration(): Promise<
    ApiResult<{ allowed_extensions: string[] }>
> {
    return request('/supported-units', { method: 'GET' });
}

export function fetchExperiments(
    datasetId: number,
): Promise<ApiResult<{ experiments: ExperimentSummary[] }>> {
    return request(`/${datasetId}/experiments`, { method: 'GET' });
}

export function fetchObservations(
    datasetId: number,
    isothermId: number,
    offset = 0,
    limit = 100,
): Promise<ApiResult<ObservationPage>> {
    return request(
        `/${datasetId}/experiments/${isothermId}/observations?offset=${offset}&limit=${limit}`,
        { method: 'GET' },
    );
}

export function deleteDataset(datasetId: number): Promise<ApiResult<null>> {
    return request(`/${datasetId}`, { method: 'DELETE' });
}

export function renameDataset(
    datasetId: number,
    newName: string,
): Promise<ApiResult<{ dataset: DatasetSummary }>> {
    return request(`/${datasetId}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: newName }),
    });
}

export function updateMetadata(
    datasetId: number,
    metadata: DatasetMetadata,
): Promise<ApiResult<{ dataset: DatasetSummary }>> {
    return request(`/${datasetId}/metadata`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metadata),
    });
}
