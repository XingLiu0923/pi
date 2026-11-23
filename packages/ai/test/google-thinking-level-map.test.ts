import type { GenerateContentParameters } from "@google/genai";
import { describe, expect, it, vi } from "vitest";
import { streamSimple as streamSimpleGoogle } from "../src/api/google-generative-ai.ts";
import { resolveGoogleThinkingLevel } from "../src/api/google-shared.ts";
import { streamSimple as streamSimpleVertex } from "../src/api/google-vertex.ts";
import type {
	FetchFunction,
	Model,
	SimpleStreamOptions,
	ThinkingBudgets,
	ThinkingLevel,
	ThinkingLevelMap,
} from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const context = normalizeContext({
	messages: [{ role: "user", content: "Hello", timestamp: 0 }],
});

function googleModel(id: string, thinkingLevelMap: ThinkingLevelMap): Model<"google-generative-ai"> {
	return {
		id,
		name: id,
		api: "google-generative-ai",
		provider: "test-google",
		baseUrl: "https://example.invalid/v1beta",
		reasoning: true,
		thinkingLevelMap,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};
}

function vertexModel(id: string, thinkingLevelMap: ThinkingLevelMap): Model<"google-vertex"> {
	return {
		id,
		name: id,
		api: "google-vertex",
		provider: "test-vertex",
		baseUrl: "https://example.invalid/v1",
		reasoning: true,
		thinkingLevelMap,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
	};
}

async function captureGooglePayload(
	model: Model<"google-generative-ai">,
	reasoning?: ThinkingLevel,
	thinkingBudgets?: ThinkingBudgets,
): Promise<GenerateContentParameters> {
	let payload: GenerateContentParameters | undefined;
	const result = await streamSimpleGoogle(model, context, {
		apiKey: "test",
		reasoning,
		thinkingBudgets,
		onPayload: (request) => {
			payload = request as GenerateContentParameters;
			throw new Error("payload captured");
		},
	}).result();

	expect(result.errorMessage).toContain("payload captured");
	if (!payload) throw new Error("Google payload was not captured");
	return payload;
}

async function captureVertexPayload(
	model: Model<"google-vertex">,
	reasoning?: ThinkingLevel,
	thinkingBudgets?: ThinkingBudgets,
): Promise<GenerateContentParameters> {
	let payload: GenerateContentParameters | undefined;
	const result = await streamSimpleVertex(model, context, {
		apiKey: "test",
		reasoning,
		thinkingBudgets,
		onPayload: (request) => {
			payload = request as GenerateContentParameters;
			throw new Error("payload captured");
		},
	}).result();

	expect(result.errorMessage).toContain("payload captured");
	if (!payload) throw new Error("Vertex payload was not captured");
	return payload;
}

const googleAdapters = [
	{
		name: "Google Generative AI",
		streamSimple: (id: string, thinkingLevelMap: ThinkingLevelMap, options: SimpleStreamOptions) =>
			streamSimpleGoogle(googleModel(id, thinkingLevelMap), context, options),
		capture: (id: string, thinkingLevelMap: ThinkingLevelMap, reasoning?: ThinkingLevel) =>
			captureGooglePayload(googleModel(id, thinkingLevelMap), reasoning),
	},
	{
		name: "Google Vertex",
		streamSimple: (id: string, thinkingLevelMap: ThinkingLevelMap, options: SimpleStreamOptions) =>
			streamSimpleVertex(vertexModel(id, thinkingLevelMap), context, options),
		capture: (id: string, thinkingLevelMap: ThinkingLevelMap, reasoning?: ThinkingLevel) =>
			captureVertexPayload(vertexModel(id, thinkingLevelMap), reasoning),
	},
] as const;

describe("Google thinking level maps", () => {
	it("exhaustively resolves supported logical levels and mapping values", () => {
		const defaultExpectations = {
			minimal: "minimal",
			low: "low",
			medium: "medium",
			high: "high",
		} as const satisfies Partial<Record<ThinkingLevel, ThinkingLevel>>;
		for (const [level, expected] of Object.entries(defaultExpectations)) {
			expect(resolveGoogleThinkingLevel(googleModel("gemini-3.7-flash", {}), level as ThinkingLevel)).toBe(expected);
		}

		const mappedExpectations = {
			minimal: "minimal",
			low: "low",
			medium: "medium",
			high: "high",
			MINIMAL: "minimal",
			LOW: "low",
			MEDIUM: "medium",
			HIGH: "high",
		} as const;
		for (const [mapped, expected] of Object.entries(mappedExpectations)) {
			const model = googleModel("gemini-3.7-flash", { high: mapped, xhigh: mapped, max: mapped });
			expect(resolveGoogleThinkingLevel(model, "high")).toBe(expected);
			expect(resolveGoogleThinkingLevel(model, "xhigh")).toBe(expected);
			expect(resolveGoogleThinkingLevel(model, "max")).toBe(expected);
		}

		const invalidModel = googleModel("gemini-3.7-flash", { xhigh: "extreme" });
		expect(() => resolveGoogleThinkingLevel(invalidModel, "xhigh")).toThrow(
			"Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> extreme",
		);
		expect(() => resolveGoogleThinkingLevel(googleModel("gemini-3.7-flash", {}), "max")).toThrow(
			"Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: max -> undefined",
		);
		expect(() => resolveGoogleThinkingLevel(googleModel("gemini-3.7-flash", {}), "xhigh")).toThrow(
			"Unsupported Google thinking level mapping for test-google/gemini-3.7-flash: xhigh -> undefined",
		);
	});

	// Regression test for https://github.com/earendil-works/pi/issues/12184 (also affects Neon, #12391).
	describe.each(googleAdapters)("$name native XHIGH", ({ streamSimple }) => {
		it.each([
			{ id: "gemini-4-neon", mapped: "xhigh" },
			{ id: "gemini-4-neon", mapped: "XHIGH" },
			{ id: "gemini-4-hydrogen", mapped: "xhigh" },
			{ id: "gemini-4-hydrogen", mapped: "XHIGH" },
		])("serializes $id with xhigh mapped to $mapped", async ({ id, mapped }) => {
			const thinkingLevelMap: ThinkingLevelMap = {
				off: null,
				minimal: null,
				low: "low",
				medium: "medium",
				high: "high",
				xhigh: mapped,
				max: null,
			};
			const fetch = vi.fn<FetchFunction>(
				async () =>
					new Response(
						`data: ${JSON.stringify({ candidates: [{ content: { role: "model", parts: [{ text: "Hello" }] }, finishReason: "STOP" }] })}\n\n`,
						{ headers: { "content-type": "text/event-stream" } },
					),
			);
			const result = await streamSimple(id, thinkingLevelMap, {
				apiKey: "test",
				reasoning: "xhigh",
				fetch,
			}).result();

			expect(result.stopReason).toBe("stop");
			expect(fetch).toHaveBeenCalledTimes(1);
			expect(JSON.parse(String(fetch.mock.calls[0][1]?.body))).toMatchObject({
				generationConfig: { thinkingConfig: { includeThoughts: true, thinkingLevel: "XHIGH" } },
			});
		});
	});

	// #12184: upgrading the SDK must not move preview models or aliases to the v1 default.
	it("keeps the Gemini beta endpoint when a model does not override the base URL", async () => {
		const model = { ...googleModel("gemini-flash-latest", {}), baseUrl: "" };
		const fetch = vi.fn<FetchFunction>(
			async () =>
				new Response(`data: ${JSON.stringify({ candidates: [{ finishReason: "STOP" }] })}\n\n`, {
					headers: { "content-type": "text/event-stream" },
				}),
		);
		const result = await streamSimpleGoogle(model, context, { apiKey: "test", reasoning: "high", fetch }).result();

		expect(result.stopReason).toBe("stop");
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(String(fetch.mock.calls[0][0])).toContain("https://generativelanguage.googleapis.com/v1beta/");
	});

	// Regression test for https://github.com/earendil-works/pi/issues/9455
	it.each(googleAdapters)("uses the lowest supported $name level when reasoning is omitted", async ({ capture }) => {
		const payload = await capture("gemini-3.8-flash", {
			off: null,
			minimal: null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: null,
			max: null,
		});

		expect(payload.config?.thinkingConfig).toEqual({ thinkingLevel: "LOW" });
	});

	it.each(googleAdapters)("preserves native medium effort for Gemini 3.1 Pro on $name", async ({ capture }) => {
		const payload = await capture(
			"gemini-3.1-pro-preview",
			{ off: null, minimal: null, low: "low", medium: "medium", high: "high", xhigh: null, max: null },
			"medium",
		);

		expect(payload.config?.thinkingConfig).toEqual({ includeThoughts: true, thinkingLevel: "MEDIUM" });
	});

	it.each(googleAdapters)("disables Gemini 2.5 thinking when reasoning is omitted on $name", async ({ capture }) => {
		const payload = await capture("gemini-2.5-flash", {});

		expect(payload.config?.thinkingConfig).toEqual({ thinkingBudget: 0 });
	});

	// Regression test for https://github.com/earendil-works/pi/issues/10890
	it.each(googleAdapters)("uses thinking levels for Gemini 4 models on $name", async ({ capture }) => {
		const payload = await capture("gemini-4-argon", {}, "medium");

		expect(payload.config?.thinkingConfig).toEqual({ includeThoughts: true, thinkingLevel: "MEDIUM" });
	});

	it.each(["xhigh", "max"] as const)("maps Google Generative AI %s to a supported level", async (reasoning) => {
		const payload = await captureGooglePayload(
			googleModel("gemini-3.7-flash", { xhigh: "high", max: "high" }),
			reasoning,
		);

		expect(payload).toMatchObject({ config: { thinkingConfig: { includeThoughts: true, thinkingLevel: "HIGH" } } });
	});

	it("honors uppercase provider values for standard Google Generative AI levels", async () => {
		const payload = await captureGooglePayload(googleModel("gemini-3.7-flash", { high: "LOW" }), "high");

		expect(payload).toMatchObject({ config: { thinkingConfig: { thinkingLevel: "LOW" } } });
	});

	it("uses mapped Google Generative AI levels for token budgets", async () => {
		const payload = await captureGooglePayload(googleModel("gemini-2.5-flash", { xhigh: "high" }), "xhigh", {
			high: 1234,
		});

		expect(payload).toMatchObject({ config: { thinkingConfig: { thinkingBudget: 1234 } } });
	});

	it("maps Google Vertex extended levels", async () => {
		const payload = await captureVertexPayload(vertexModel("gemini-3.7-flash", { xhigh: "high" }), "xhigh");

		expect(payload).toMatchObject({ config: { thinkingConfig: { includeThoughts: true, thinkingLevel: "HIGH" } } });
	});

	it("uses mapped Google Vertex levels for token budgets", async () => {
		const payload = await captureVertexPayload(vertexModel("gemini-2.5-flash", { max: "high" }), "max", {
			high: 4321,
		});

		expect(payload).toMatchObject({ config: { thinkingConfig: { thinkingBudget: 4321 } } });
	});
});
