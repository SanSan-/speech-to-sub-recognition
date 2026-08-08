import { expect, test } from "@playwright/test";

const SETTINGS_KEY = "speechToSubSettingsV1";
const CONTROL_LOG_LINE = "E2E-КОНТРОЛЬ-SSE";
const PATHS = [
  "D:\\E2E\\Лекция 01.mp4",
  "D:\\E2E\\Разбор алгоритма №2.mkv",
  "D:\\E2E\\Ошибка дорожки 03.wav",
];

function fileCard(page, name) {
  return page.locator(".file-card").filter({ hasText: name });
}

async function controlLineCount(page) {
  const text = (await page.locator("#logConsole").textContent()) ?? "";
  return text.split(CONTROL_LOG_LINE).length - 1;
}

async function e2eState(request) {
  const response = await request.get("/__e2e__/state");
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test("SQLite сохраняет batch при reload, cancel и retry", async ({ page, request }) => {
  await page.goto("/");

  const backend = page.locator("#backend");
  const aligner = page.locator("#aligner");
  await backend.selectOption("faster-whisper");
  await aligner.selectOption("qwen3-forced-aligner");
  await expect(backend).toHaveValue("faster-whisper");
  await expect(aligner).toHaveValue("qwen3-forced-aligner");
  await expect(page.locator("#modelPath")).toHaveValue(/whisper-large-v3-ct2$/);
  await expect(page.locator("#alignerModelPath")).toHaveValue(
    /Qwen3-ForcedAligner-0\.6B$/,
  );
  await backend.selectOption("parakeet-tdt-v3");
  await expect(aligner).toHaveValue("qwen3-forced-aligner");
  await backend.selectOption("faster-whisper");

  const language = page.getByLabel("Язык распознавания");
  const device = page.getByLabel("Устройство");
  const cancel = page.getByRole("button", { name: "Отменить" });
  const retry = page.getByRole("button", { name: "Повторить ошибки" });
  await language.selectOption("ru");
  await device.selectOption("cpu");

  await expect
    .poll(() =>
      page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? "null"), SETTINGS_KEY),
    )
    .toMatchObject({ language: "ru", device: "cpu" });

  await page.reload();
  await expect(language).toHaveValue("ru");
  await expect(device).toHaveValue("cpu");

  await page.getByRole("button", { name: "Файлы" }).click();
  await expect(page.locator(".file-card")).toHaveCount(3);
  await expect(page.locator(".file-name")).toHaveText([
    "Лекция 01.mp4",
    "Разбор алгоритма №2.mkv",
    "Ошибка дорожки 03.wav",
  ]);

  await page.getByRole("button", { name: "Запустить" }).click();
  await expect(fileCard(page, "Лекция 01.mp4")).toHaveAttribute("data-status", "done");
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "transcribing",
  );
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toContainText("42%");
  await expect(page.locator("#logConsole")).toContainText(CONTROL_LOG_LINE);
  await expect.poll(() => controlLineCount(page)).toBe(1);

  await page.reload();

  await expect(page.locator(".file-card")).toHaveCount(3);
  await expect(fileCard(page, "Лекция 01.mp4")).toHaveAttribute("data-status", "done");
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "transcribing",
  );
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toContainText("42%");
  await expect(language).toHaveValue("ru");
  await expect(device).toHaveValue("cpu");
  await expect(language).toBeDisabled();
  await expect(device).toBeDisabled();
  await expect(cancel).toBeEnabled();
  await expect.poll(() => controlLineCount(page)).toBe(1);

  const activeSnapshot = await request.get("/api/active-job");
  expect(activeSnapshot.ok()).toBeTruthy();
  const originalJobId = (await activeSnapshot.json()).job_id;
  expect(await e2eState(request)).toMatchObject({
    entered: true,
    cancel_seen: false,
    calls: [PATHS],
  });

  await cancel.click();
  await expect(cancel).toBeDisabled();
  await expect.poll(async () => (await e2eState(request)).cancel_seen).toBe(true);
  await expect(page.locator("#terminalStatus")).toContainText("частично");
  await expect(fileCard(page, "Лекция 01.mp4")).toHaveAttribute("data-status", "done");
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "cancelled",
  );
  await expect(fileCard(page, "Ошибка дорожки 03.wav")).toHaveAttribute(
    "data-status",
    "cancelled",
  );
  await expect(retry).toBeEnabled();
  await expect(language).toBeEnabled();
  await expect(device).toBeEnabled();
  await expect.poll(() => controlLineCount(page)).toBe(1);

  const restarted = await request.post("/__e2e__/restart");
  expect(restarted.ok()).toBeTruthy();
  expect(await restarted.json()).toMatchObject({
    job_id: originalJobId,
    status: "partial",
    terminal: true,
  });
  await page.reload();

  await expect(page.locator(".file-card")).toHaveCount(3);
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "cancelled",
  );
  await expect(retry).toBeEnabled();
  await expect(language).toHaveValue("ru");
  await expect(device).toHaveValue("cpu");
  await expect.poll(() => controlLineCount(page)).toBe(1);

  await retry.click();
  await expect(page.locator(".file-card")).toHaveCount(2);
  await expect(page.locator(".file-name")).toHaveText([
    "Разбор алгоритма №2.mkv",
    "Ошибка дорожки 03.wav",
  ]);
  await expect(language).toBeDisabled();
  await expect(device).toBeDisabled();
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toContainText("42%");
  await expect.poll(async () => (await e2eState(request)).retry_entered).toBe(true);
  expect(await e2eState(request)).toMatchObject({ calls: [PATHS, PATHS.slice(1)] });

  const release = await request.post("/__e2e__/release");
  expect(release.ok()).toBeTruthy();
  await expect(page.locator("#terminalStatus")).toContainText("частично");
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "done",
  );
  await expect(fileCard(page, "Ошибка дорожки 03.wav")).toHaveAttribute(
    "data-status",
    "error",
  );
  await expect(retry).toBeEnabled();
  await expect.poll(() => controlLineCount(page)).toBe(1);

  const history = await request.get("/api/jobs?limit=2");
  expect(history.ok()).toBeTruthy();
  const jobs = (await history.json()).jobs;
  expect(jobs).toHaveLength(2);
  expect(jobs[0].retry_of).toBe(originalJobId);
  expect(jobs[1].job_id).toBe(originalJobId);
});
