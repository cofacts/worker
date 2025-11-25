import {
	WorkflowEntrypoint,
	WorkflowEvent,
	WorkflowStep,
} from "cloudflare:workers";
import { UrlResolverWorkflow } from "./workflows/url-resolver";
import { ArticleClassifierWorkflow } from "./workflows/article-classifier";

export { UrlResolverWorkflow, ArticleClassifierWorkflow };

async function authenticate(req: Request, env: Env): Promise<boolean> {
	const serviceTokenId = env.SERVICE_TOKEN_ID;
	const serviceTokenSecret = env.SERVICE_TOKEN_SECRET;

	// If secrets are not set, fail closed (or open for dev? Better fail closed)
	if (!serviceTokenId || !serviceTokenSecret) {
		console.error("Service Token secrets are not set in environment");
		return false;
	}

	const clientId = req.headers.get("CF-Access-Client-Id");
	const clientSecret = req.headers.get("CF-Access-Client-Secret");

	return clientId === serviceTokenId && clientSecret === serviceTokenSecret;
}

export default {
	async fetch(req: Request, env: Env): Promise<Response> {
		const url = new URL(req.url);
		const path = url.pathname;

		// Health check or root
		if (path === "/" || path === "/health") {
			return Response.json({ status: "ok" });
		}

		// Authentication check for /workflows/*
		if (path.startsWith("/workflows/")) {
			const isAuthenticated = await authenticate(req, env);
			if (!isAuthenticated) {
				return Response.json({ error: "Unauthorized" }, { status: 401 });
			}
		}

		// Route: POST /workflows/:name
		// Trigger a workflow
		if (req.method === "POST" && path.startsWith("/workflows/")) {
			const match = path.match(/\/workflows\/([^\/]+)$/);
			if (match) {
				const workflowName = match[1];
				let workflowInstance;

				try {
					const payload = await req.json();

					if (workflowName === "url-resolver") {
						workflowInstance = await env.URL_RESOLVER.create({ params: payload });
					} else if (workflowName === "article-classifier") {
						workflowInstance = await env.ARTICLE_CLASSIFIER.create({ params: payload });
					} else {
						return Response.json({ error: "Workflow not found" }, { status: 404 });
					}

					return Response.json({
						id: workflowInstance.id,
						status: "started",
						timestamp: new Date().toISOString(),
					});
				} catch (e: any) {
					return Response.json({ error: e.message }, { status: 400 });
				}
			}
		}

		// Route: GET /workflows/:name/:id
		// Get workflow status
		if (req.method === "GET" && path.startsWith("/workflows/")) {
			const match = path.match(/\/workflows\/([^\/]+)\/([^\/]+)$/);
			if (match) {
				const workflowName = match[1];
				const instanceId = match[2];
				let workflowBinding;

				if (workflowName === "url-resolver") {
					workflowBinding = env.URL_RESOLVER;
				} else if (workflowName === "article-classifier") {
					workflowBinding = env.ARTICLE_CLASSIFIER;
				} else {
					return Response.json({ error: "Workflow not found" }, { status: 404 });
				}

				try {
					const instance = await workflowBinding.get(instanceId);
					const status = await instance.status();

					// If completed, we might want to include the output if available in status
					// Status type usually has output if completed

					return Response.json({
						id: instance.id,
						status: status.status,
						output: status.output,
						error: status.error,
					});
				} catch (e: any) {
					return Response.json({ error: "Instance not found or error fetching status" }, { status: 404 });
				}
			}
		}

		return Response.json({ error: "Not Found" }, { status: 404 });
	},
};
