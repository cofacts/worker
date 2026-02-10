import {
  WorkflowEntrypoint,
  WorkflowEvent,
  WorkflowStep,
} from "cloudflare:workers";
import { GoogleGenAI } from "@google/genai";

type ArticleClassifierParams = {
  articleId: string;
  dryRun?: boolean;
};

type CofactsCategory = {
  id: string;
  title: string;
  description: string;
};

export class ArticleClassifierWorkflow extends WorkflowEntrypoint<Env, ArticleClassifierParams> {
  async run(event: WorkflowEvent<ArticleClassifierParams>, step: WorkflowStep) {
    const { articleId, dryRun = false } = event.payload;

    if (!articleId) {
      throw new Error("Article ID is required");
    }

    // Step 1: Fetch article text and categories
    const { articleText, categories } = await step.do("fetch-article-and-categories", async () => {
      const query = `
				query GetArticleAndCategories($id: String!) {
					GetArticle(id: $id) {
						text
					}
					ListCategories(first: 50) {
						edges {
							node {
								id
								title
								description
							}
						}
					}
				}
			`;

      const response = await fetch(this.env.RUMORS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          variables: { id: articleId },
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch article data: ${response.statusText}`);
      }

      const data = await response.json() as any;

      if (!data.data?.GetArticle) {
        throw new Error(`Article not found: ${articleId}`);
      }

      return {
        articleText: data.data.GetArticle.text,
        categories: data.data.ListCategories.edges.map((edge: any) => edge.node) as CofactsCategory[],
      };
    });

    // Step 2: Classify with Gemini
    const classification = await step.do("classify-article", async () => {
      const ai = new GoogleGenAI({ apiKey: this.env.GEMINI_API_KEY });

      const categoryList = categories.map(cat => `## ${cat.id}: ${cat.title}\n ${cat.description}`).join('\n\n');

      const prompt = `You are a rumor classification expert. Classify the given text into one or more categories from Cofacts.

# Available categories
${categoryList}

# Instructions
Respond with a JSON object containing:
- categoryIds: array of category IDs (from the IDs provided above) that match the article. If none apply, return an empty array.
- reasoning: brief explanation in Chinese for the classification choices

# Rumor Article Text
${articleText}`;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
        config: {
          responseSchema: {
            type: "object" as const,
            properties: {
              categoryIds: {
                type: "array" as const,
                items: { type: "string" as const },
              },
              reasoning: { type: "string" as const },
            },
            required: ["categoryIds", "reasoning"],
          },
          responseMimeType: "application/json",
        }
      });

      const content = response.text;
      if (!content) {
        throw new Error("Empty response from Gemini");
      }

      return JSON.parse(content) as { categoryIds: string[], reasoning: string };
    });

    // Step 3: Save results to Cofacts API (skipped if dryRun is true)
    if (dryRun) {
      return {
        articleId,
        classification,
        dryRun: true,
        message: "Classification completed but skipped saving due to dryRun=true",
      };
    }

    const saveResult = await step.do("save-classification", async () => {
      const results = [];
      for (const categoryId of classification.categoryIds) {
        const response = await fetch(this.env.RUMORS_API_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-app-id": "RUMORS_WORKER",
          },
          body: JSON.stringify({
            query: `
							mutation($articleId: String!, $categoryId: String!) {
								CreateArticleCategory(articleId: $articleId, categoryId: $categoryId) {
									articleId
								}
							}
						`,
            variables: {
              articleId,
              categoryId
            },
          }),
        });

        const json = await response.json() as any;
        results.push(json);
      }

      return {
        articleId,
        saved: results,
        reasoning: classification.reasoning
      };
    });

    return saveResult;
  }
}
