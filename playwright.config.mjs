import { defineConfig } from "@playwright/test";

const port = Number.parseInt(process.env.SPEECH_TO_SUB_E2E_PORT ?? "17862", 10);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "*.spec.mjs",
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  outputDir: "./tests/e2e/.artifacts",
  reporter: "list",
  use: {
    baseURL,
    headless: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `.\\.venv\\Scripts\\python.exe -B tests\\e2e\\fake_web_server.py --port ${port}`,
    url: `${baseURL}/api/health`,
    reuseExistingServer: false,
    timeout: 30_000,
    env: {
      PYTHONDONTWRITEBYTECODE: "1",
      WEB_JOB_DB: "tests/e2e/.artifacts/jobs.sqlite3",
    },
  },
});
