import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((promiseResolve, promiseReject) => {
        resolve = promiseResolve;
        reject = promiseReject;
    });
    return { promise, resolve, reject };
}

const capabilities = (machineLearning: boolean) => ({
    version: '3.0.0',
    features: {
        datasets: true,
        nist: true,
        fitting: true,
        machine_learning: machineLearning,
        training: machineLearning,
        checkpoints: machineLearning,
    },
});

async function loadService() {
    vi.resetModules();
    return await import('./system.service');
}

describe('system service capability discovery', () => {
    const fetchMock = vi.fn();

    beforeEach(() => {
        fetchMock.mockReset();
        vi.stubGlobal('fetch', fetchMock);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it('retries capability discovery after a transient failure', async () => {
        fetchMock
            .mockRejectedValueOnce(new Error('backend starting'))
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(true),
            });
        const { machineLearningAvailable } = await loadService();

        expect(await machineLearningAvailable()).toBe(false);
        expect(await machineLearningAvailable()).toBe(true);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it('reuses a successful capability response without duplicate requests', async () => {
        fetchMock.mockResolvedValue({
            ok: true,
            json: async () => capabilities(false),
        });
        const { fetchApplicationCapabilities } = await loadService();

        const first = await fetchApplicationCapabilities();
        const second = await fetchApplicationCapabilities();

        expect(first.data?.features.machine_learning).toBe(false);
        expect(second.data).toEqual(first.data);
        expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it('refreshes a previously cached capability response when requested', async () => {
        fetchMock
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(false),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(true),
            });
        const { fetchApplicationCapabilities } = await loadService();

        expect((await fetchApplicationCapabilities()).data?.features.machine_learning).toBe(false);
        expect((await fetchApplicationCapabilities(true)).data?.features.machine_learning).toBe(true);
        expect((await fetchApplicationCapabilities()).data?.features.machine_learning).toBe(true);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it('coalesces concurrent forced capability refreshes', async () => {
        const response = deferred<{ ok: boolean; json: () => Promise<unknown> }>();
        fetchMock.mockReturnValue(response.promise);
        const { fetchApplicationCapabilities } = await loadService();

        const first = fetchApplicationCapabilities(true);
        const second = fetchApplicationCapabilities(true);

        expect(second).toBe(first);
        expect(fetchMock).toHaveBeenCalledTimes(1);

        response.resolve({
            ok: true,
            json: async () => capabilities(true),
        });
        expect((await first).data?.features.machine_learning).toBe(true);
        expect((await second).data?.features.machine_learning).toBe(true);
    });

    it('allows a forced capability refresh to retry after failure', async () => {
        const firstResponse = deferred<{ ok: boolean; status: number; json: () => Promise<unknown> }>();
        fetchMock
            .mockReturnValueOnce(firstResponse.promise)
            .mockResolvedValueOnce({
                ok: true,
                json: async () => capabilities(true),
            });
        const { fetchApplicationCapabilities } = await loadService();

        const first = fetchApplicationCapabilities(true);
        const concurrent = fetchApplicationCapabilities(true);
        expect(concurrent).toBe(first);
        expect(fetchMock).toHaveBeenCalledTimes(1);

        firstResponse.resolve({
            ok: false,
            status: 503,
            json: async () => ({ detail: 'backend unavailable' }),
        });
        expect((await first).data).toBeNull();
        expect((await concurrent).data).toBeNull();

        const retry = await fetchApplicationCapabilities(true);
        expect(retry.data?.features.machine_learning).toBe(true);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});
