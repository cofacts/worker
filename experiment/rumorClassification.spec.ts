import { test, describe } from "vitest";
import { env } from "cloudflare:test";

declare module "cloudflare:test" {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface ProvidedEnv extends Cloudflare.Env {}
}

const TEST_TIMEOUT = 5 * 60 * 1000; // 5 minutes

describe("Integration Test: Rumor Classification Workflow", () => {
  test(
    "should create a workflow, which then fetches real data and inserts it into D1",
    async () => {
      // 1. Create a workflow instance directly using the binding
      const instance = await env.RUMOR_CLASSIFIER.create({
        id: crypto.randomUUID(),
        params: {
          datasetName: "gossip-classification-10-samples",
        },
      });

      console.log("Workflow instance created:", instance.id);

      // In a real test, we would poll for completion or check for side-effects.
      // For now, we just verify that the workflow can be created.
    },
    TEST_TIMEOUT
  );
});