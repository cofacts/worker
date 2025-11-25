import {
  WorkflowEntrypoint,
  WorkflowEvent,
  WorkflowStep,
} from "cloudflare:workers";

type UrlResolverParams = {
  url: string;
};

type UrlResolverResult = {
  title: string;
  description: string;
  image: string;
  url: string;
  error?: string;
};

export class UrlResolverWorkflow extends WorkflowEntrypoint<Env, UrlResolverParams> {
  async run(event: WorkflowEvent<UrlResolverParams>, step: WorkflowStep) {
    const { url } = event.payload;

    if (!url) {
      throw new Error("URL is required");
    }

    const result = await step.do("fetch-and-parse-url", async () => {
      try {
        const response = await fetch(url, {
          headers: {
            "User-Agent": "Cofacts-Worker/1.0",
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch URL: ${response.status} ${response.statusText}`);
        }

        const html = await response.text();

        // Simple regex-based parsing to avoid heavy DOM parser dependencies
        const getMetaContent = (prop: string) => {
          const match = html.match(new RegExp(`<meta\\s+(?:property|name)=["']${prop}["']\\s+content=["'](.*?)["']`, "i"));
          return match ? match[1] : "";
        };

        const getTitle = () => {
          const ogTitle = getMetaContent("og:title");
          if (ogTitle) return ogTitle;
          const titleMatch = html.match(/<title>(.*?)<\/title>/i);
          return titleMatch ? titleMatch[1] : "";
        };

        return {
          title: getTitle(),
          description: getMetaContent("og:description") || getMetaContent("description"),
          image: getMetaContent("og:image"),
          url: getMetaContent("og:url") || url,
        };
      } catch (error: any) {
        return {
          title: "",
          description: "",
          image: "",
          url: url,
          error: error.message,
        };
      }
    });

    return result;
  }
}
