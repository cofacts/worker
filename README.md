# Cofacts Worker

Cloudflare worker for Cofacts' asynchronous tasks.

## Workflows

### Rumor Classification

This workflow classifies rumors using OpenAI's batch API and logs the results to Langfuse for evaluation.

#### Local Testing

To test the rumor classification workflow locally:

1.  **Install dependencies:**
    ```bash
    npm install
    ```

2.  **Set up environment variables:**

    Create a `.dev.vars` file in the root of the project and add the following environment variables:

    ```
    LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
    LANGFUSE_SECRET_KEY=your_langfuse_secret_key
    OPENAI_API_KEY=your_openai_api_key
    ```

    You can also set `LANGFUSE_HOST` if you are using a self-hosted Langfuse instance.

3.  **Start the development server:**

    The `--test-scheduled` flag is required to test the `scheduled` handler locally.

    ```bash
    npx wrangler dev --test-scheduled
    ```

4.  **Trigger the workflow:**

    Open a new terminal and run the following command to trigger the scheduled event:

    ```bash
    curl -X POST "http://localhost:8787/__scheduled?cron=*+*+*+*+*"
    ```

5.  **Monitor the workflow:**

    The workflow will start and you can monitor its progress in the `wrangler dev` terminal. The workflow will poll OpenAI's batch API until the classification is complete.

    You can also monitor the traces and experiments in your Langfuse project.
