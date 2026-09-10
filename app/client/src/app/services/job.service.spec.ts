import { describe, expect, it } from 'vitest';
import { normalizePollingIntervalSeconds, resolvePollingIntervalMs } from './job.service';

describe('job polling interval helpers', () => {
    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, null, undefined])(
        'rejects unsafe polling interval %s',
        (value) => {
            expect(normalizePollingIntervalSeconds(value)).toBeNull();
            expect(resolvePollingIntervalMs(value)).toBeNull();
        },
    );

    it('keeps positive intervals and converts them to milliseconds', () => {
        expect(normalizePollingIntervalSeconds(0.5)).toBe(0.5);
        expect(resolvePollingIntervalMs(0.5)).toBe(500);
    });
});
