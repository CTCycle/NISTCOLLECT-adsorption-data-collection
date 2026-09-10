import type { JobStartResponse, JobStatusResponse } from '../models/job.model';
import { API_BASE_URL } from '../core/config/api-base-url';
import { extractErrorMessage, fetchWithTimeout, HTTP_TIMEOUT } from './http-timeout.service';

export type JobStartResult = {
    jobId: string | null;
    pollInterval?: number;
    error: string | null;
};

export const normalizePollingIntervalSeconds = (intervalSeconds: number | null | undefined): number | null => {
    if (typeof intervalSeconds !== 'number' || !Number.isFinite(intervalSeconds) || intervalSeconds <= 0) {
        return null;
    }

    return intervalSeconds;
};

export const resolvePollingIntervalMs = (intervalSeconds: number | null | undefined): number | null => {
    const normalizedSeconds = normalizePollingIntervalSeconds(intervalSeconds);
    if (normalizedSeconds === null) {
        return null;
    }

    return normalizedSeconds * 1000;
};

export async function pollJobStatus(endpoint: string, jobId: string): Promise<{ data: JobStatusResponse | null; error: string | null }> {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}${endpoint}/jobs/${jobId}`, { method: 'GET' }, HTTP_TIMEOUT);
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { data: null, error: extractErrorMessage(response, data) };
        }

        const result = await response.json().catch(() => null);
        if (!result || typeof result !== 'object') {
            return { data: null, error: 'Invalid job status response.' };
        }
        return { data: result as JobStatusResponse, error: null };
    } catch (error) {
        return { data: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}

export async function pollJobUntilTerminal(
    endpoint: string,
    jobId: string,
    pollInterval?: number,
    onProgress?: (status: JobStatusResponse) => void
): Promise<{ data: JobStatusResponse | null; error: string | null }> {
    while (true) {
        const result = await pollJobStatus(endpoint, jobId);
        if (result.error || !result.data) {
            return result;
        }
        const status = result.data;

        onProgress?.(status);

        if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
            return { data: status, error: null };
        }

        const intervalMs = resolvePollingIntervalMs(status.poll_interval ?? pollInterval);
        if (intervalMs === null) {
            return { data: null, error: 'Job response omitted a polling interval.' };
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
}
export async function startJob(
    endpoint: string,
    payload: unknown = {},
    timeout: number = HTTP_TIMEOUT
): Promise<JobStartResult> {
    try {
        const response = await fetchWithTimeout(
            `${API_BASE_URL}${endpoint}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            },
            timeout
        );

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            return { jobId: null, error: extractErrorMessage(response, data) };
        }

        const result = (await response.json()) as JobStartResponse;
        return { jobId: result.job_id, pollInterval: result.poll_interval, error: null };
    } catch (error) {
        return { jobId: null, error: error instanceof Error ? error.message : 'An unknown error occurred.' };
    }
}
