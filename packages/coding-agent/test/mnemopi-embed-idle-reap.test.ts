/**
 * The mnemopi embeddings subprocess (`__omp_worker_mnemopi_embed`) used to
 * live for the whole session: spawned on the first recall, SIGKILLed only on
 * dispose or after a wedged request. With `BAAI/bge-base-en-v1.5` resident
 * each idle worker holds ~700 MB of private anonymous memory, so a desk with a
 * dozen open sessions pays ~10 GB for workers that answered their last embed
 * hours ago. The client now reaps the worker after an idle window; the next
 * request respawns it and the embed message re-initializes the model (the
 * respawn contract pinned by the #3031 tests). These tests drive the client
 * with fake workers so no fastembed/onnxruntime is needed.
 */
import { describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { MnemopiEmbedClient, type MnemopiEmbedWorkerHandle } from "@oh-my-pi/pi-coding-agent/mnemopi/embed-client";
import type {
	MnemopiEmbedWorkerInbound,
	MnemopiEmbedWorkerOutbound,
} from "@oh-my-pi/pi-coding-agent/mnemopi/embed-protocol";

const IDLE_MS = 1_000;

interface FakeWorkerState {
	spawns: number;
	terminated: number;
	/** Set while the fake is holding back an embed reply; call to deliver it. */
	release: (() => void) | undefined;
}

/**
 * Fake worker that answers `init` immediately and answers `embed` either
 * immediately or, when `holdEmbeds` is set, only once the test calls
 * `state.release()`.
 */
function fakeEmbedWorker(state: FakeWorkerState, holdEmbeds = false): () => MnemopiEmbedWorkerHandle {
	return () => {
		state.spawns += 1;
		let handler: ((message: MnemopiEmbedWorkerOutbound) => void) | undefined;
		return {
			send(message: MnemopiEmbedWorkerInbound) {
				if (message.type === "ping") queueMicrotask(() => handler?.({ type: "pong", id: message.id }));
				else if (message.type === "init") queueMicrotask(() => handler?.({ type: "ready", id: message.id }));
				else if (message.type === "embed") {
					const reply = () => handler?.({ type: "vectors", id: message.id, vectors: [[1, 2, 3]] });
					if (holdEmbeds) state.release = reply;
					else queueMicrotask(reply);
				}
			},
			onMessage(next) {
				handler = next;
				return () => {
					if (handler === next) handler = undefined;
				};
			},
			onError() {
				return () => {};
			},
			ref() {},
			unref() {},
			async terminate() {
				state.terminated += 1;
				handler = undefined;
			},
		};
	};
}

async function drain(model: { embed(texts: string[]): AsyncIterable<number[][]> }, text: string): Promise<number[][]> {
	for await (const vectors of model.embed([text])) return vectors;
	throw new Error("no vectors");
}

describe("mnemopi embed worker is reaped while idle", () => {
	it("terminates the worker after the idle window and respawns it on the next embed", async () => {
		vi.useFakeTimers();
		const state: FakeWorkerState = { spawns: 0, terminated: 0, release: undefined };
		const client = new MnemopiEmbedClient(fakeEmbedWorker(state), 60_000, IDLE_MS);
		try {
			const model = await client.initialize("fast-bge-base-en-v1.5", "/tmp/cache");
			expect(model).not.toBeNull();
			expect(await drain(model!, "first")).toEqual([[1, 2, 3]]);
			expect(state.spawns).toBe(1);

			vi.advanceTimersByTime(IDLE_MS - 1);
			expect(state.terminated).toBe(0);
			vi.advanceTimersByTime(1);
			expect(state.terminated).toBe(1);

			// The cached wrapper still works: a fresh worker is spawned and the
			// embed carries (model, cacheDir) so the child self-initializes.
			expect(await drain(model!, "second")).toEqual([[1, 2, 3]]);
			expect(state.spawns).toBe(2);
		} finally {
			await client.terminate();
			vi.useRealTimers();
		}
	});

	it("does not reap a worker while a request is in flight longer than the idle window", async () => {
		vi.useFakeTimers();
		const state: FakeWorkerState = { spawns: 0, terminated: 0, release: undefined };
		const client = new MnemopiEmbedClient(fakeEmbedWorker(state, true), 60_000, IDLE_MS);
		try {
			const model = await client.initialize("fast-bge-base-en-v1.5", "/tmp/cache");
			const embedding = drain(model!, "slow");
			// Let the embed request reach the fake worker before advancing time.
			await Promise.resolve();
			expect(state.release).toBeDefined();

			vi.advanceTimersByTime(IDLE_MS * 10);
			expect(state.terminated).toBe(0);

			state.release?.();
			expect(await embedding).toEqual([[1, 2, 3]]);
			expect(state.spawns).toBe(1);

			// Idle accounting starts once the request settles, not when it was sent.
			vi.advanceTimersByTime(IDLE_MS - 1);
			expect(state.terminated).toBe(0);
			vi.advanceTimersByTime(1);
			expect(state.terminated).toBe(1);
		} finally {
			await client.terminate();
			vi.useRealTimers();
		}
	});

	it("does not keep a print-mode process alive waiting for the idle window", async () => {
		// A ref'd idle timer would pin `omp -p` for the whole window after its
		// last recall (the #12067 failure class in reverse). The assertion is
		// the child's own exit, so this needs a real process and real time:
		// the idle window is set far beyond the budget allowed for the exit.
		const repoRoot = path.resolve(import.meta.dir, "../../..");
		const script = `
			const { MnemopiEmbedClient } = await import("@oh-my-pi/pi-coding-agent/mnemopi/embed-client");
			let handler;
			const client = new MnemopiEmbedClient(() => ({
				send(message) {
					queueMicrotask(() => handler?.(message.type === "init"
						? { type: "ready", id: message.id }
						: { type: "vectors", id: message.id, vectors: [[0]] }));
				},
				onMessage(next) { handler = next; return () => {}; },
				onError() { return () => {}; },
				ref() {}, unref() {},
				async terminate() {},
			}), 60_000, 60_000);
			const model = await client.initialize("fast-bge-base-en-v1.5", "/tmp/cache");
			for await (const _ of model.embed(["bye"])) {}
		`;
		const start = Date.now();
		const proc = Bun.spawn([process.execPath, "-e", script], {
			cwd: repoRoot,
			stdout: "pipe",
			stderr: "pipe",
		});
		const [stdout, stderr, exitCode] = await Promise.all([
			new Response(proc.stdout).text(),
			new Response(proc.stderr).text(),
			proc.exited,
		]);
		expect(`${stdout}${stderr}`).toBe("");
		expect(exitCode).toBe(0);
		expect(Date.now() - start).toBeLessThan(20_000);
	}, 30_000);
});
