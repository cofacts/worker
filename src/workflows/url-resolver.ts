import {
  WorkflowEntrypoint,
  WorkflowEvent,
  WorkflowStep,
} from "cloudflare:workers";
import { GoogleGenAI } from "@google/genai";
import puppeteer from "@cloudflare/puppeteer";

type UrlResolverParams = {
  url: string;
};

type UrlResolverResult = {
  url: string;
  canonical: string;
  title: string;
  summary: string;
  topImageUrl: string;
  status: 'success' | 'error';
  error?: string;
};

export class UrlResolverWorkflow extends WorkflowEntrypoint<Env, UrlResolverParams> {
  async run(event: WorkflowEvent<UrlResolverParams>, step: WorkflowStep) {
    const { url } = event.payload;

    if (!url) {
      throw new Error("URL is required");
    }

    // Step 1: Use Gemini with Google Search tool to fetch URL content
    const geminiResult = await step.do("gemini-resolve", {
      retries: {
        limit: 3,
        delay: "5 seconds",
        backoff: "exponential",
      },
    }, async () => {
      const ai = new GoogleGenAI({ apiKey: this.env.GEMINI_API_KEY });

      const prompt = `Please analyze the following URL and extract:
1. Canonical URL (the final URL after redirects and removing tracking parameters)
2. Page title
3. Main content summary (the full article text if possible, cleaned of navigation/ads)
4. Representative image URL (OG image or main news photo)

URL: ${url}

Respond in JSON format with fields: canonical, title, summary, topImageUrl.`;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt,
        config: {
          tools: [{ googleSearch: {} }],
          responseMimeType: "application/json",
        },
      });

      const responseText = response.text;
      if (!responseText) {
        throw new Error("Empty response from Gemini");
      }
      return JSON.parse(responseText) as {
        canonical: string;
        title: string;
        summary: string;
        topImageUrl: string;
      };
    });

    // Step 2: Check if Gemini result is sufficient, if not, use Browser Rendering
    const finalResult = await step.do("finalize-result", async () => {
      // If summary is too short or fields are missing, try browser rendering
      const isPoorResult = !geminiResult.summary || geminiResult.summary.length < 50 || !geminiResult.title;

      if (isPoorResult) {
        // Fallback to Browser Rendering if BROWSER binding is available
        if (this.env.BROWSER) {
          try {
            const browser = await puppeteer.launch(this.env.BROWSER);
            const page = await browser.newPage();
            await page.goto(url, { waitUntil: "networkidle2" });

            const pageData = await page.evaluate(() => {
              const getMeta = (prop: string) => {
                // @ts-expect-error - DOM APIs available in browser context
                const el = document.querySelector(`meta[property="${prop}"], meta[name="${prop}"]`);
                return el ? el.getAttribute("content") : "";
              };

              return {
                // @ts-expect-error - DOM APIs available in browser context
                title: document.title || getMeta("og:title"),
                // @ts-expect-error - DOM APIs available in browser context
                canonical: getMeta("og:url") || window.location.href,
                // @ts-expect-error - DOM APIs available in browser context
                summary: (document.querySelector("article")?.innerText || document.body.innerText).slice(0, 2000),
                topImageUrl: getMeta("og:image"),
              };
            });

            await browser.close();

            // Use browser data if it's better
            return {
              url,
              canonical: pageData.canonical || geminiResult.canonical || url,
              title: pageData.title || geminiResult.title,
              summary: pageData.summary || geminiResult.summary,
              topImageUrl: pageData.topImageUrl || geminiResult.topImageUrl,
              status: 'success' as const,
            };
          } catch (e: any) {
            // If browser rendering fails, just stick with gemini result
            return {
              url,
              ...geminiResult,
              status: 'success' as const,
            };
          }
        }
      }

      return {
        url,
        ...geminiResult,
        status: 'success' as const,
      };
    });

    return finalResult;
  }
}
