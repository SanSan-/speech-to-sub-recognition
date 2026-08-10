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

async function measureDesktopLayout(page) {
  return page.evaluate(() => {
    const app = document.querySelector(".app");
    const main = document.querySelector(".main-grid");
    const files = document.querySelector(".panel-files");
    const settings = document.querySelector(".panel-settings");
    const fileList = document.querySelector("#fileList");
    const settingsForm = document.querySelector("#settingsForm");
    const logs = document.querySelector(".panel-logs");
    const actions = document.querySelector(".job-actions");
    const appRect = app.getBoundingClientRect();
    const mainRect = main.getBoundingClientRect();
    const filesRect = files.getBoundingClientRect();
    const settingsRect = settings.getBoundingClientRect();
    const logsRect = logs.getBoundingClientRect();
    const actionsRect = actions.getBoundingClientRect();
    return {
      appHeight: appRect.height,
      workspaceHeight: mainRect.height,
      equalPanelHeight: Math.abs(filesRect.height - settingsRect.height) < 1,
      fileOverflow: getComputedStyle(fileList).overflowY,
      settingsOverflow: getComputedStyle(settingsForm).overflowY,
      fileScrollable: fileList.scrollHeight > fileList.clientHeight,
      settingsScrollable: settingsForm.scrollHeight > settingsForm.clientHeight,
      logsFollowMain: main.nextElementSibling === logs && logsRect.top >= mainRect.bottom,
      logsGap: Math.round(logsRect.top - mainRect.bottom),
      bottomGap: Math.round(window.innerHeight - logsRect.bottom),
      logsVisible: logsRect.bottom <= window.innerHeight,
      actionsVisible: actionsRect.top >= 0 && actionsRect.bottom <= window.innerHeight,
      pageFitsViewport: document.documentElement.scrollHeight <= window.innerHeight + 1,
    };
  });
}

test("рабочие панели растягиваются, настройки сгруппированы и мобильный поток свободен", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");

  const alignmentSettings = page.locator("#alignmentSettings");
  const srtSettings = page.locator("#srtSettings");
  const longFormSettings = page.locator("#longFormSettings");
  const advancedSettings = page.locator("#advancedSettings");
  await expect(alignmentSettings).not.toHaveAttribute("open", "");
  await expect(srtSettings).toHaveAttribute("open", "");
  await expect(longFormSettings).not.toHaveAttribute("open", "");
  await expect(advancedSettings).not.toHaveAttribute("open", "");

  await alignmentSettings.locator("summary").click();
  await longFormSettings.locator("summary").click();
  await advancedSettings.locator("summary").click();
  await expect(alignmentSettings).toHaveAttribute("open", "");
  await expect(longFormSettings).toHaveAttribute("open", "");
  await expect(advancedSettings).toHaveAttribute("open", "");
  await expect(page.locator('label[for="force"] .field-hint')).toContainText(
    "Обойти кеш готового SRT и распознавания",
  );

  await page.getByLabel("Допуск длины строки SRT").fill("7");
  await page.getByLabel("Окно длинной записи, сек.").fill("420");
  await page.getByLabel("Сохранить нормализованный FLAC").check();
  await expect
    .poll(() =>
      page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? "null"), SETTINGS_KEY),
    )
    .toMatchObject({
      line_length_gap: 7,
      long_form_window_seconds: 420,
      keep_audio: true,
    });

  await page.getByRole("button", { name: "Файлы" }).click();
  await expect(page.locator(".file-card")).toHaveCount(3);
  await page.locator("#fileList").evaluate((list) => {
    const template = list.querySelector(".file-card");
    if (!template) return;
    for (let index = 0; index < 3; index += 1) {
      list.append(template.cloneNode(true));
    }
  });

  const desktopLayout = await measureDesktopLayout(page);
  expect(desktopLayout).toMatchObject({
    equalPanelHeight: true,
    fileOverflow: "auto",
    settingsOverflow: "auto",
    fileScrollable: true,
    settingsScrollable: true,
    logsFollowMain: true,
    logsVisible: true,
    actionsVisible: true,
  });
  expect(desktopLayout.workspaceHeight).toBeGreaterThanOrEqual(430);
  expect(desktopLayout.logsGap).toBeLessThanOrEqual(20);
  await expect(page.locator("body")).not.toContainText("ASR backend");
  await expect(page.locator("body")).not.toContainText("Beam size");
  await expect(page.locator("body")).not.toContainText("Long-form");

  const tallLayouts = [];
  for (const height of [1200, 1600]) {
    await page.setViewportSize({ width: 1440, height });
    const layout = await measureDesktopLayout(page);
    expect(layout).toMatchObject({
      appHeight: height,
      equalPanelHeight: true,
      fileOverflow: "auto",
      settingsOverflow: "auto",
      logsFollowMain: true,
      logsVisible: true,
      actionsVisible: true,
      pageFitsViewport: true,
    });
    expect(layout.workspaceHeight).toBeGreaterThanOrEqual(430);
    expect(layout.logsGap).toBeLessThanOrEqual(20);
    expect(layout.bottomGap).toBeGreaterThanOrEqual(23);
    expect(layout.bottomGap).toBeLessThanOrEqual(25);
    tallLayouts.push(layout);
  }
  expect(tallLayouts[1].workspaceHeight - tallLayouts[0].workspaceHeight).toBeGreaterThan(390);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload();
  await expect(page.locator("#advancedSettings")).not.toHaveAttribute("open", "");
  await page.locator("#advancedSettings > summary").click();
  await expect(page.getByLabel("Сохранить нормализованный FLAC")).toBeChecked();
  await page.locator("#longFormSettings > summary").click();
  await expect(page.getByLabel("Окно длинной записи, сек.")).toHaveValue("420");
  await expect(page.getByLabel("Допуск длины строки SRT")).toHaveValue("7");

  const mobileLayout = await page.evaluate(() => {
    const fileList = document.querySelector("#fileList");
    const settingsForm = document.querySelector("#settingsForm");
    const files = document.querySelector(".panel-files");
    const settings = document.querySelector(".panel-settings");
    return {
      fileOverflow: getComputedStyle(fileList).overflowY,
      settingsOverflow: getComputedStyle(settingsForm).overflowY,
      fileFitsContent: fileList.scrollHeight <= fileList.clientHeight + 1,
      settingsFitsContent: settingsForm.scrollHeight <= settingsForm.clientHeight + 1,
      ordinaryFlow: settings.getBoundingClientRect().top >= files.getBoundingClientRect().bottom,
    };
  });
  expect(mobileLayout).toEqual({
    fileOverflow: "visible",
    settingsOverflow: "visible",
    fileFitsContent: true,
    settingsFitsContent: true,
    ordinaryFlow: true,
  });
});

test("облачный режим требует несохраняемого согласия и скрывает локальные параметры", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Файлы" }).click();

  const backend = page.locator("#backend");
  const consent = page.getByLabel("Разрешить передачу аудио в OpenAI");
  const start = page.getByRole("button", { name: "Запустить" });
  await backend.selectOption("openai-api");

  await expect(page.locator("#brandSubtitle")).toHaveText("Облачное пакетное распознавание речи в SRT");
  await expect(page.locator("#settingsModeHint")).toContainText("Аудио передаётся в OpenAI");
  await expect(page.locator("#cloudSettings")).toBeVisible();
  await expect(page.locator("#openAiModel")).toHaveValue("whisper-1");
  await expect(page.locator("#openAiModel option")).toHaveCount(1);
  await expect(page.locator("#localModelSettings")).toBeHidden();
  await expect(page.locator("#modelPath")).toBeDisabled();
  await expect(page.locator("#deviceSettings")).toBeHidden();
  await expect(page.locator("#workerPythonPath")).toBeDisabled();
  await expect(page.locator("#quantizationEnabled")).toBeDisabled();
  await expect(page.locator("#autoDownloadModel")).toBeDisabled();
  await expect(consent).not.toBeChecked();
  await expect(start).toBeDisabled();

  await consent.check();
  await expect(start).toBeEnabled();
  const stored = await page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? "{}"), SETTINGS_KEY);
  expect(stored).not.toHaveProperty("allow_cloud_processing");

  await backend.selectOption("faster-whisper");
  await backend.selectOption("openai-api");
  await expect(consent).not.toBeChecked();
  await expect(start).toBeDisabled();
});

test("SQLite сохраняет batch при reload, cancel и retry", async ({ page, request }) => {
  await page.goto("/");

  const backend = page.locator("#backend");
  await page.locator("#alignmentSettings > summary").click();
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
  const maxCharsPerLine = page.getByLabel("Базовая длина строки SRT");
  const lineLengthGap = page.getByLabel("Допуск длины строки SRT");
  const maxCps = page.getByLabel("Ориентир скорости чтения, символов/с");
  const cancel = page.getByRole("button", { name: "Отменить" });
  const retry = page.getByRole("button", { name: "Повторить ошибки" });
  await language.selectOption("ru");
  await device.selectOption("cpu");
  await maxCharsPerLine.fill("40");
  await lineLengthGap.fill("6");
  await maxCps.fill("16.5");

  await expect
    .poll(() =>
      page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? "null"), SETTINGS_KEY),
    )
    .toMatchObject({
      language: "ru",
      device: "cpu",
      max_chars_per_line: 40,
      line_length_gap: 6,
      max_cps: 16.5,
    });

  await page.reload();
  await expect(language).toHaveValue("ru");
  await expect(device).toHaveValue("cpu");
  await expect(maxCharsPerLine).toHaveValue("40");
  await expect(lineLengthGap).toHaveValue("6");
  await expect(maxCps).toHaveValue("16.5");

  await page.getByRole("button", { name: "Файлы" }).click();
  await expect(page.locator(".file-card")).toHaveCount(3);
  await expect(page.locator(".file-name")).toHaveText([
    "Лекция 01.mp4",
    "Разбор алгоритма №2.mkv",
    "Ошибка дорожки 03.wav",
  ]);

  await page.locator("#advancedSettings > summary").click();
  const force = page.getByLabel("Перезаписать целевые результаты");
  await force.check();
  await expect
    .poll(async () => {
      const settings = (await e2eState(request)).build_settings;
      return settings.at(-1)?.force;
    })
    .toBe(true);
  await expect(page.locator('.file-card[data-status="cached"]')).toHaveCount(0);
  const storedBeforeStart = await page.evaluate(
    (key) => JSON.parse(localStorage.getItem(key) ?? "{}"),
    SETTINGS_KEY,
  );
  expect(storedBeforeStart).not.toHaveProperty("force");

  await page.getByRole("button", { name: "Запустить" }).click();
  await expect(force).not.toBeChecked();
  await expect
    .poll(async () => (await e2eState(request)).process_settings[0]?.force)
    .toBe(true);
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
  await page.locator("#advancedSettings > summary").click();
  await expect(force).not.toBeChecked();
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
  await expect
    .poll(async () => (await e2eState(request)).process_settings[1]?.force)
    .toBe(false);

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
