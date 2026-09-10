import { API_BASE_URL } from '../core/config/api-base-url';
import type { FittingConfiguration } from '../models/fitting.model';
import type { TrainingConfiguration } from '../models/training.model';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

export interface ApplicationCapabilities {
    version: string;
    features: { datasets: boolean; nist: boolean; fitting: boolean; machine_learning: boolean; training: boolean; checkpoints: boolean; };
}
export interface ServiceHealth { service: string; version: string; state: string; }
export interface ServiceResult<T> { data: T | null; error: string | null; }
const asRecord = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' ? value as Record<string, unknown> : null;
const readJson = async <T>(url: string): Promise<ServiceResult<T>> => {
    try {
        const response = await fetchWithTimeout(url, { method: 'GET' }, HTTP_TIMEOUT);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) return { data: null, error: extractErrorMessage(response, body) };
        if (!asRecord(body)) return { data: null, error: `Invalid response from ${url}.` };
        return { data: body as T, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
};
let cachedCapabilities: ApplicationCapabilities | null = null;
let capabilitiesRequest: Promise<ServiceResult<ApplicationCapabilities>> | null = null;
export const fetchApplicationCapabilities = (refresh = false): Promise<ServiceResult<ApplicationCapabilities>> => {
    if (capabilitiesRequest !== null) return capabilitiesRequest;
    if (!refresh && cachedCapabilities !== null) {
        return Promise.resolve({ data: cachedCapabilities, error: null });
    }

    const request = readJson<ApplicationCapabilities>(`${API_BASE_URL}/system/capabilities`).then((result) => {
        if (result.data !== null) cachedCapabilities = result.data;
        if (capabilitiesRequest === request) capabilitiesRequest = null;
        return result;
    });
    capabilitiesRequest = request;
    return request;
};
export const machineLearningAvailable = async (): Promise<boolean> => (await fetchApplicationCapabilities()).data?.features.machine_learning === true;
export const fetchFittingConfiguration = (): Promise<ServiceResult<FittingConfiguration>> => readJson<FittingConfiguration>(`${API_BASE_URL}/system/configuration`);
export const fetchTrainingConfiguration = (): Promise<ServiceResult<TrainingConfiguration>> => readJson<TrainingConfiguration>(`${API_BASE_URL}/training/configuration`);
export const fetchBackendReadiness = (): Promise<ServiceResult<ServiceHealth>> => readJson<ServiceHealth>('/health/ready');
