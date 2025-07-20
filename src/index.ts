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

type Message = {
	id: string;
	text: string;
	metadata?: Record<string, any>;
};

type CofactsCategory = {
	id: string;
	title: string;
	description: string;
};

type ClassificationResult = {
	id: string;
	classification: {
		categories: string[];
		reasoning?: string;
	};
	usage?: any;
};

function createClassificationRequest(message: Message, categories: CofactsCategory[]) {
	const categoryList = categories.map(cat => `## ${cat.title}\n ${cat.description}`).join('\n\n');
	return {
		custom_id: message.id,
		method: "POST",
		url: "/v1/chat/completions",
		body: {
			model: "gpt-4o-mini",
			messages: [
				{
					role: "system",
					content: `You are a rumor classification expert. Classify the given text into one or more categories from Cofacts.

# Available categories
${categoryList}

# Instructions
Respond with a JSON object containing:
- categories: array of exact category titles from the list above. If none of the categories apply, return an empty array. Provide only the titles, not the descriptions.
- reasoning: brief explanation for the classification choices`
				},
				{
					role: "user",
					content: `Classify this rumor article: ${message.text}`
				}
			],
			response_format: { type: "json_object" },
			temperature: 0.1,
		}
	};
}

// Calculate multi-class accuracy function
function calculateMultiClassScore(expected: string[], predicted: string[]): number {
	const expectedSet = new Set(expected);
	const predictedSet = new Set(predicted);

	// Perfect match gets 1.0 - use Set intersection to check equality
	const intersection = expectedSet.intersection(predictedSet);
	if (expectedSet.size === predictedSet.size && intersection.size === expectedSet.size) {
		return 1.0;
	}

	// Calculate size difference
	const difference = Math.abs(expectedSet.size - predictedSet.size);

	// If difference is exactly 1 (one extra or one missing), get 0.5
	if (difference === 1) {
		// Check if intersection covers the smaller set completely
		const minSize = Math.min(expectedSet.size, predictedSet.size);
		if (intersection.size === minSize) {
			return 0.5;
		}
	}

	return 0.0;
}

export class RumorClassificationWorkflow extends WorkflowEntrypoint<Env, RumorClassificationParams> {
	SKIPPED_CATEGORY_IDS = new Set([
		'nj2n7nEBrIRcahlY-gpc', // 無意義 🚧
		'nT2n7nEBrIRcahlY6QqF', // 有意義但不包含在以上標籤 🚧
		'lj2m7nEBrIRcahlY6Ao_', // 基本人權問題 🚧
		'kz3c7XEBrIRcahlYxAp6', // 性少數與愛滋病 🚧
		'oD2o7nEBrIRcahlYFgpm', // 只有網址其他資訊不足 🚧
	]);

	IGNORE_DATASET_ITEMS_WITH_CATEGORY = 'oD2o7nEBrIRcahlYFgpm'; // 只有網址其他資訊不足 🚧

	async run(event: WorkflowEvent<RumorClassificationParams>, step: WorkflowStep) {
		const datasetName = event.payload.datasetName || this.env.DATASET_NAME;

		// Initialize clients
		const langfuse = new Langfuse({
			publicKey: this.env.LANGFUSE_PUBLIC_KEY,
			secretKey: this.env.LANGFUSE_SECRET_KEY,
			baseUrl: this.env.LANGFUSE_HOST,
		});

		const openai = new OpenAI({
			apiKey: this.env.OPENAI_API_KEY,
		});

		// Step 1: Load categories and dataset in parallel
		const [categories, messagesToCategorize] = await Promise.all([
			step.do("load-cofacts-categories", async () => {
				const response = await fetch("https://api.cofacts.tw/graphql", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						query: "query ListCategories { ListCategories(first: 50) { edges { node { id title description } } } }"
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
				const allCategories = data.data.ListCategories.edges.map(edge => edge.node);
				return allCategories.filter(cat => !this.SKIPPED_CATEGORY_IDS.has(cat.id));
			}),
			step.do("load-messages-to-categorize", async () => {
				try {
					const dataset = await langfuse.getDataset(datasetName);

					const items = dataset.items.slice(0, 20);
					const filteredItems = items.filter((item: any) => {
						const expected = (item.expectedOutput as string[]) || [];
						return !expected.includes(this.IGNORE_DATASET_ITEMS_WITH_CATEGORY);
					});
					return filteredItems.map((item: any): Message => ({
						id: item.id,
						text: item.input?.text || item.input,
						metadata: item.metadata,
					}));
				} catch (error) {
					throw new Error(`Failed to load dataset '${datasetName}': ${error}`);
				}
			})
		]);

		// Step 2: Upload batch to OpenAI LLM service
		const batchUpload = await step.do("upload-batch-to-openai", async () => {
			const batchRequests = messagesToCategorize.map((item: Message) =>
				createClassificationRequest(item, categories)
			);

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
					limit: 720, // 24 hours / 2 minutes = 720 retries
					delay: "2 minutes",
					backoff: "constant",
				},
				timeout: "25 hours", // Slightly longer than 24h to account for processing
			},
			async () => {
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
						const usage = response.response.body.usage;

						results.push({
							id: response.custom_id,
							classification,
							usage,
						});
					}

					return {
						batch,
						results,
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
			// Reload dataset to get original items with .link() method
			const dataset = await langfuse.getDataset(datasetName);
			const originalDatasetItems = dataset.items.slice(0, 20); // Same 20 items

			// Create category mappings
			const categoryIdToName: Record<string, string> = {};
			const categoryNameToId: Record<string, string> = {};
			categories.forEach(cat => {
				categoryIdToName[cat.id] = cat.title;
				categoryNameToId[cat.title] = cat.id;
			});

			const runName = `batch-classification-${Date.now()}`;
			const evaluationResults = [];
			let totalScore = 0;
			let totalPredictions = 0;

			// Create ID-to-item mappings for O(1) lookup
			const messageById = new Map<string, Message>();
			messagesToCategorize.forEach(item => messageById.set(item.id, item));

			const datasetItemById = new Map<string, any>();
			originalDatasetItems.forEach(item => datasetItemById.set(item.id, item));

			for (const result of batchResult.results) {
				const message = messageById.get(result.id);
				const datasetItem = datasetItemById.get(result.id);
				if (!message || !datasetItem) continue;

				// Get expected categories from original dataset item
				const rawExpectedCategories = (datasetItem.expectedOutput as string[]) || []; // Always a list of category IDs
				const expectedCategories = rawExpectedCategories.filter(id => !this.SKIPPED_CATEGORY_IDS.has(id));

				// Convert predicted category titles to category IDs for comparison
				const predictedCategories = result.classification.categories.map((title: string) => categoryNameToId[title] || title).filter(Boolean);
				const score = calculateMultiClassScore(expectedCategories, predictedCategories);

				totalScore += score;
				totalPredictions++;

				// Get category names for metadata
				const expectedCategoryNames = expectedCategories.map(id => categoryIdToName[id] || id);
				const predictedCategoryNames = result.classification.categories; // These are already titles from step 4
				const isCorrect = score === 1.0;

				// Create trace for individual prediction
				const trace = langfuse.trace({
					name: "rumor-classification",
					input: {
						text: message.text,
						expectedCategories: expectedCategoryNames,
						expectedCategoryIds: expectedCategories,
					},
					output: {
						predictedCategories: predictedCategoryNames,
						predictedCategoryIds: predictedCategories,
						reasoning: result.classification.reasoning,
					},
					metadata: {
						datasetName,
						batchId: batchResult.batch.id,
						score: score,
						isCorrect: isCorrect,
						datasetItemId: message.id,
						...message.metadata,
					},
				});

				// Add generation span for the LLM call
				const requestObj = createClassificationRequest(message, categories);
				const { created_at, completed_at, failed_at, expired_at } = batchResult.batch;
				const endTime = completed_at || failed_at || expired_at;

				trace.generation({
					name: "openai-classification",
					startTime: new Date(created_at * 1000),
					endTime: endTime ? new Date(endTime * 1000) : undefined,
					model: "gpt-4o-mini",
					input: {
						messages: requestObj.body.messages
					},
					output: {
						categories: predictedCategoryNames,
						categoryIds: predictedCategories,
						reasoning: result.classification.reasoning,
					},
					usage: result.usage,
				});

				// Add score for evaluation
				trace.score({
					name: "multi-class-accuracy",
					value: score,
					comment: score === 1.0 ? "Perfect prediction" :
						score === 0.5 ? "Partial match (1 difference)" :
						`Expected: ${expectedCategoryNames.join(', ')}, Got: ${predictedCategoryNames.join(', ')}`,
				});

				// Link trace to dataset item for experiment tracking
				await datasetItem.link(trace, runName, {
					description: `Batch classification experiment using OpenAI gpt-4o-mini`,
					metadata: {
						batchId: batchResult.batch.id,
						model: "gpt-4o-mini",
						totalItems: messagesToCategorize.length
					},
				});

				evaluationResults.push({
					id: result.id,
					expected: expectedCategories,
					predicted: predictedCategories,
					score: score,
				});
			}

			await langfuse.flushAsync();

			return {
				runName,
				accuracy: totalScore / totalPredictions,
				averageScore: totalScore / totalPredictions,
				totalPredictions,
				evaluationResults,
			};
		});

		// Report workflow completion
		console.info(`🎉 Rumor Classification Workflow Completed!
- Dataset: ${datasetName}
- Run Name: ${evaluation.runName}
- Items Processed: ${messagesToCategorize.length}
- Batch ID: ${batchResult.batch.id}
- Average Score: ${evaluation.averageScore.toFixed(3)}
- Total Predictions: ${evaluation.totalPredictions}
- Categories Used: ${categories.length}`);

		return {
			datasetName,
			runName: evaluation.runName,
			itemsProcessed: messagesToCategorize.length,
			batchId: batchResult.batch.id,
			accuracy: evaluation.accuracy,
			averageScore: evaluation.averageScore,
			totalPredictions: evaluation.totalPredictions,
			categoriesUsed: categories.map(c => c.title),
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