import {
	WorkflowEntrypoint,
	WorkflowEvent,
	WorkflowStep,
} from "cloudflare:workers";
import OpenAI from "openai";
import { Langfuse } from "langfuse";

type RumorClassificationParams = {
	datasetName?: string;
	batchSize?: number;
};

type DatasetItem = {
	id: string;
	text: string;
	expectedCategory?: string;
	metadata?: Record<string, any>;
};

type CofactsCategory = {
	id: string;
	title: string;
};

type ClassificationResult = {
	id: string;
	category: string;
	confidence?: number;
	reasoning?: string;
};

export class RumorClassificationWorkflow extends WorkflowEntrypoint<Env, RumorClassificationParams> {
	async run(event: WorkflowEvent<RumorClassificationParams>, step: WorkflowStep) {
		const datasetName = event.payload.datasetName || this.env.DATASET_NAME;

		// Step 1: Load categories and dataset in parallel
		const [categories, datasetItems] = await Promise.all([
			step.do("load-cofacts-categories", async () => {
				const response = await fetch("https://api.cofacts.tw/graphql", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						query: "query ListCategories { ListCategories { edges { node { id title } } } }"
					}),
				});

				if (!response.ok) {
					throw new Error(`Failed to fetch categories: ${response.statusText}`);
				}

				const data = await response.json() as {
					data: {
						ListCategories: {
							edges: { node: CofactsCategory }[]
						}
					}
				};
				return data.data.ListCategories.edges.map(edge => edge.node);
			}),
			step.do("load-langfuse-dataset", async () => {
				const langfuse = new Langfuse({
					publicKey: this.env.LANGFUSE_PUBLIC_KEY,
					secretKey: this.env.LANGFUSE_SECRET_KEY,
					baseUrl: this.env.LANGFUSE_HOST,
				});

				try {
					const dataset = await langfuse.getDataset(datasetName);

					return dataset.items.slice(0, 20).map((item: any): DatasetItem => ({
						id: item.id,
						text: item.input?.text || item.input,
						expectedCategory: item.expectedOutput?.category || item.expectedOutput,
						metadata: item.metadata,
					}));
				} catch (error) {
					throw new Error(`Failed to load dataset '${datasetName}': ${error}`);
				}
			})
		]);

		// Step 2: Upload batch to OpenAI LLM service
		const batchUpload = await step.do("upload-batch-to-openai", async () => {
			const openai = new OpenAI({
				apiKey: this.env.OPENAI_API_KEY,
			});

			const categoryList = categories.map(cat => `- ${cat.title}`).join('\n');

			const batchRequests = datasetItems.map((item: DatasetItem) => ({
				custom_id: item.id,
				method: "POST",
				url: "/v1/chat/completions",
				body: {
					model: "gpt-4o-mini",
					messages: [
						{
							role: "system",
							content: `You are a rumor classification expert. Classify the given text into one of these categories from Cofacts:
${categoryList}

Respond with a JSON object containing:
- category: the exact category title from the list above
- confidence: confidence score (0.0-1.0)
- reasoning: brief explanation for the classification`
						},
						{
							role: "user",
							content: `Classify this rumor article: ${item.text}`
						}
					],
					response_format: { type: "json_object" },
					temperature: 0.1,
				}
			}));

			// Create JSONL content for batch upload
			const jsonlContent = batchRequests.map((req: any) => JSON.stringify(req)).join('\n');

			// Upload file for batch processing
			const file = await openai.files.create({
				file: new File([jsonlContent], 'batch_requests.jsonl', { type: 'application/jsonl' }),
				purpose: 'batch',
			});

			return {
				fileId: file.id,
				requestCount: batchRequests.length,
			};
		});

		// Step 3: Trigger OpenAI batch API
		const batchJob = await step.do("trigger-batch-api", async () => {
			const openai = new OpenAI({
				apiKey: this.env.OPENAI_API_KEY,
			});

			const batch = await openai.batches.create({
				input_file_id: batchUpload.fileId,
				endpoint: "/v1/chat/completions",
				completion_window: "24h",
			});

			return {
				batchId: batch.id,
				status: batch.status,
				createdAt: batch.created_at,
			};
		});

		// Step 4: Periodically check for batch completion
		const batchResult = await step.do(
			"poll-batch-completion",
			{
				retries: {
					limit: 1440, // 24 hours * 60 minutes
					delay: "1 minute",
					backoff: "constant",
				},
				timeout: "25 hours", // Slightly longer than 24h to account for processing
			},
			async () => {
				const openai = new OpenAI({
					apiKey: this.env.OPENAI_API_KEY,
				});

				const batch = await openai.batches.retrieve(batchJob.batchId);

				if (batch.status === "completed") {
					// Download and parse results
					const resultsFile = await openai.files.content(batch.output_file_id!);
					const resultsText = await resultsFile.text();

					const results: ClassificationResult[] = [];
					const lines = resultsText.trim().split('\n');

					for (const line of lines) {
						const response = JSON.parse(line);
						const content = response.response.body.choices[0].message.content;
						const classification = JSON.parse(content);

						results.push({
							id: response.custom_id,
							category: classification.category,
							confidence: classification.confidence,
							reasoning: classification.reasoning,
						});
					}

					return {
						batchId: batch.id,
						results,
						usage: {
							prompt_tokens: batch.request_counts?.completed || 0,
							completion_tokens: batch.request_counts?.completed || 0,
							total_tokens: batch.request_counts?.completed || 0,
						},
					};
				} else if (batch.status === "failed" || batch.status === "expired" || batch.status === "cancelled") {
					throw new Error(`Batch processing failed with status: ${batch.status}`);
				} else {
					// Still processing, retry
					throw new Error(`Batch still processing, status: ${batch.status}`);
				}
			}
		);

		// Step 5: Compare results and write to Langfuse
		const evaluation = await step.do("evaluate-and-log-results", async () => {
			const langfuse = new Langfuse({
				publicKey: this.env.LANGFUSE_PUBLIC_KEY,
				secretKey: this.env.LANGFUSE_SECRET_KEY,
				baseUrl: this.env.LANGFUSE_HOST,
			});

			// Reload dataset to get original items with .link() method
			const dataset = await langfuse.getDataset(datasetName);
			const originalDatasetItems = dataset.items.slice(0, 20); // Same 20 items

			const runName = `batch-classification-${Date.now()}`;
			const evaluationResults = [];
			let correctPredictions = 0;
			let totalPredictions = 0;

			for (const result of batchResult.results) {
				const originalItem = datasetItems.find((item: DatasetItem) => item.id === result.id);
				const datasetItem = originalDatasetItems.find((item: any) => item.id === result.id);
				if (!originalItem || !datasetItem) continue;

				const isCorrect = originalItem.expectedCategory === result.category;
				if (isCorrect) correctPredictions++;
				totalPredictions++;

				// Create trace for individual prediction
				const trace = langfuse.trace({
					name: "rumor-classification",
					input: {
						text: originalItem.text,
						expectedCategory: originalItem.expectedCategory,
					},
					output: {
						predictedCategory: result.category,
						confidence: result.confidence,
						reasoning: result.reasoning,
					},
					metadata: {
						datasetName,
						batchId: batchResult.batchId,
						correct: isCorrect,
						availableCategories: categories.map(c => c.title),
						...originalItem.metadata,
					},
				});

				// Add generation span for the LLM call
				trace.generation({
					name: "openai-classification",
					model: "gpt-4o-mini",
					input: {
						messages: [
							{
								role: "system",
								content: "You are a rumor classification expert..."
							},
							{
								role: "user",
								content: `Classify this rumor article: ${originalItem.text}`
							}
						]
					},
					output: {
						category: result.category,
						confidence: result.confidence,
						reasoning: result.reasoning,
					},
					usage: {
						promptTokens: Math.floor(batchResult.usage.prompt_tokens / batchResult.results.length),
						completionTokens: Math.floor(batchResult.usage.completion_tokens / batchResult.results.length),
						totalTokens: Math.floor(batchResult.usage.total_tokens / batchResult.results.length),
					},
				});

				// Add score for evaluation
				trace.score({
					name: "accuracy",
					value: isCorrect ? 1 : 0,
					comment: isCorrect ? "Correct prediction" : `Expected: ${originalItem.expectedCategory}, Got: ${result.category}`,
				});

				// Link trace to dataset item for experiment tracking
				await datasetItem.link(trace, runName, {
					description: `Batch classification experiment using OpenAI gpt-4o-mini`,
					metadata: {
						batchId: batchResult.batchId,
						model: "gpt-4o-mini",
						totalItems: datasetItems.length
					},
				});

				evaluationResults.push({
					id: result.id,
					expected: originalItem.expectedCategory,
					predicted: result.category,
					correct: isCorrect,
					confidence: result.confidence,
				});
			}

			await langfuse.flushAsync();

			return {
				runName,
				accuracy: correctPredictions / totalPredictions,
				correctPredictions,
				totalPredictions,
				evaluationResults,
			};
		});

		return {
			datasetName,
			runName: evaluation.runName,
			itemsProcessed: datasetItems.length,
			batchId: batchResult.batchId,
			accuracy: evaluation.accuracy,
			correctPredictions: evaluation.correctPredictions,
			totalPredictions: evaluation.totalPredictions,
			categoriesUsed: categories.map(c => c.title),
			usage: batchResult.usage,
		};
	}
}
export default {
	async scheduled(controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
		// This is a sample implementation.
		// You can customize the dataset name or other parameters as needed.
		await env.RUMOR_CLASSIFIER.create({
			id: crypto.randomUUID(),
			params: {
				datasetName: env.DATASET_NAME,
			},
		});
	},
};