import { DestroyRef, Injectable, inject } from '@angular/core';
import { Subscription, timer } from 'rxjs';
import { normalizePollingIntervalSeconds } from '../../../services/job.service';
import { getTrainingStatus } from '../../../services/training.service';
import type { TrainingStatus } from '../../../models/training.model';

const DEFAULT_TRAINING_STATUS: TrainingStatus = {
    is_training: false,
    current_epoch: 0,
    total_epochs: 0,
    progress: 0,
    metrics: {},
    history: [],
    log: [],
};

@Injectable({ providedIn: 'root' })
export class TrainingStatusPollingService {
    private readonly destroyRef = inject(DestroyRef);
    private pollSubscription: Subscription | null = null;
    private pollIntervalSeconds: number | null = null;
    private wasTraining = false;
    private statusRequestInFlight = false;
    private pollingGeneration = 0;

    constructor() {
        this.destroyRef.onDestroy(() => this.stopPolling());
    }

    startPolling(
        intervalSeconds: number | undefined,
        onStatus: (status: TrainingStatus) => void,
        onError: (error: string | null) => void,
        onTrainingEnded?: () => void
    ): void {
        this.stopPolling();
        const normalizedInterval = normalizePollingIntervalSeconds(intervalSeconds);
        if (normalizedInterval === null) {
            onError('Training response omitted a polling interval.');
            return;
        }
        this.pollIntervalSeconds = normalizedInterval;
        const generation = this.pollingGeneration;
        this.pollSubscription = timer(0, normalizedInterval * 1000).subscribe(() => {
            void this.checkStatus(onStatus, onError, onTrainingEnded, generation);
        });
    }

    stopPolling(): void {
        this.pollingGeneration += 1;
        this.pollSubscription?.unsubscribe();
        this.pollSubscription = null;
        this.pollIntervalSeconds = null;
    }

    async checkStatus(
        onStatus: (status: TrainingStatus) => void,
        onError: (error: string | null) => void,
        onTrainingEnded?: () => void,
        generation = this.pollingGeneration,
    ): Promise<void> {
        if (generation !== this.pollingGeneration || this.statusRequestInFlight) {
            return;
        }

        this.statusRequestInFlight = true;
        try {
            const result = await getTrainingStatus();
            if (generation !== this.pollingGeneration) {
                return;
            }

            if (result.error || !result.data) {
                onError(result.error || 'Training response omitted status data.');
                return;
            }

            onError(null);
            const status = result.data;
            const wasTraining = this.wasTraining;
            this.wasTraining = status.is_training;
            const nextStatus: TrainingStatus = {
                is_training: status.is_training,
                current_epoch: status.current_epoch,
                total_epochs: status.total_epochs,
                progress: status.progress,
                metrics: status.metrics || {},
                history: status.history || [],
                log: status.log || [],
                poll_interval: status.poll_interval,
            };
            onStatus(nextStatus);
            if (generation !== this.pollingGeneration) {
                return;
            }

            const nextInterval = normalizePollingIntervalSeconds(status.poll_interval);
            if (status.is_training) {
                if (nextInterval === null) {
                    this.stopPolling();
                    onError('Training response omitted a polling interval.');
                    return;
                }
                if (nextInterval !== this.pollIntervalSeconds) {
                    this.startPolling(nextInterval, onStatus, onError, onTrainingEnded);
                    return;
                }
            } else {
                this.stopPolling();
                if (wasTraining) {
                    onTrainingEnded?.();
                }
            }
        } finally {
            this.statusRequestInFlight = false;
        }
    }

    createDefaultStatus(): TrainingStatus {
        return { ...DEFAULT_TRAINING_STATUS, metrics: {}, history: [], log: [] };
    }
}
