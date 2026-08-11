import { expect, test } from "@playwright/test";

const SETTINGS_KEY = "speechToSubSettingsV1";
const CONTROL_LOG_LINE = "E2E-КОНТРОЛЬ-SSE";
const FOLDER_DISCOVERY_LOG = "Сбор каталога завершён: найдено медиафайлов — 2.";
const FOLDER_PROBING_LOGS = [
  "Проверка аудиопотоков: 0 из 2.",
  "Проверка аудиопотоков: 1 из 2.",
  "Проверка аудиопотоков: 2 из 2.",
];
const LARGE_ITEM_COUNT = 10_000;
const E2E_PORT = Number.parseInt(process.env.SPEECH_TO_SUB_E2E_PORT ?? "17862", 10);
const LOCAL_ORIGIN_HEADERS = Object.freeze({
  Origin: `http://127.0.0.1:${E2E_PORT}`,
});
const PATHS = [
  "D:\\E2E\\Лекция 01.mp4",
  "D:\\E2E\\Разбор алгоритма №2.mkv",
  "D:\\E2E\\Ошибка дорожки 03.wav",
];

function fileCard(page, name) {
  return page.locator(".file-card").filter({
    has: page.getByText(name, { exact: true }),
  });
}

function fileDuration(page, name) {
  return fileCard(page, name).locator(".file-duration");
}

async function controlLineCount(page) {
  const text = (await page.locator("#logConsole").textContent()) ?? "";
  return text.split(CONTROL_LOG_LINE).length - 1;
}

async function logTextCount(page, value) {
  const text = (await page.locator("#logConsole").textContent()) ?? "";
  return text.split(value).length - 1;
}

async function e2eState(request) {
  const response = await request.get("/__e2e__/state");
  expect(response.ok()).toBeTruthy();
  return response.json();
}

async function releaseFolderPreparation(request) {
  const response = await request.post("/__e2e__/release-preparation", {
    headers: LOCAL_ORIGIN_HEADERS,
  });
  expect(response.ok()).toBeTruthy();
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

test("папка рекурсивно готовится с ранним прогрессом и без скрытого обновления", async ({ page, request }) => {
  await page.addInitScript((key) => {
    localStorage.setItem(key, JSON.stringify({ recursive: false }));
  }, SETTINGS_KEY);

  const requestPayloads = { pick: [], refresh: [] };
  let preparationStatusRequests = 0;
  page.on("request", (browserRequest) => {
    const pathname = new URL(browserRequest.url()).pathname;
    if (browserRequest.method() === "GET" && pathname === "/api/preparation-status") {
      preparationStatusRequests += 1;
      return;
    }
    if (browserRequest.method() !== "POST") return;
    if (pathname === "/api/pick") {
      requestPayloads.pick.push(browserRequest.postDataJSON());
    } else if (pathname === "/api/refresh") {
      requestPayloads.refresh.push(browserRequest.postDataJSON());
    }
  });

  await page.goto("/");
  await expect(page.locator("#recursive")).toBeChecked();
  await expect(page.locator("#recursive")).toBeDisabled();

  await page.getByRole("button", { name: "Папка" }).click();
  try {
    await expect
      .poll(async () => (await e2eState(request)).folder_preparation_entered)
      .toBe(true);
    await expect(page.locator("#logConsole")).toContainText(
      "Открываю системный диалог выбора папки",
    );
    await expect(page.locator("#logConsole")).toContainText(FOLDER_DISCOVERY_LOG);
    await expect(page.locator("#logConsole")).toContainText(FOLDER_PROBING_LOGS[0]);
    await expect(page.locator(".file-card")).toHaveCount(0);
    await expect(page.locator("#pickerFileBtn")).toBeDisabled();
    await expect(page.locator("#pickerFolderBtn")).toBeDisabled();
    await expect(page.locator("#refreshBtn")).toBeDisabled();
    await expect(page.locator("#transcribeBtn")).toBeDisabled();
    await expect(page.locator("#unloadBtn")).toBeDisabled();
    await expect(page.locator("#language")).toBeDisabled();

    const statusRequestsBeforeRepeat = preparationStatusRequests;
    await expect
      .poll(() => preparationStatusRequests)
      .toBeGreaterThanOrEqual(statusRequestsBeforeRepeat + 2);
    await expect.poll(() => logTextCount(page, FOLDER_DISCOVERY_LOG)).toBe(1);
    await expect.poll(() => logTextCount(page, FOLDER_PROBING_LOGS[0])).toBe(1);
  } finally {
    await releaseFolderPreparation(request);
  }

  await expect(page.locator(".file-card")).toHaveCount(2);
  await expect(page.locator(".file-name")).toHaveText([
    "Лекция верхнего уровня.mp4",
    "Глубокий разбор №4.mkv",
  ]);
  await expect(page.locator("#selectedPath")).toContainText("Каталог Юникод");
  for (const message of [FOLDER_DISCOVERY_LOG, ...FOLDER_PROBING_LOGS]) {
    await expect(page.locator("#logConsole")).toContainText(message);
    await expect.poll(() => logTextCount(page, message)).toBe(1);
  }

  expect(requestPayloads.pick).toHaveLength(1);
  expect(requestPayloads.pick[0]).toMatchObject({
    kind: "folder",
    settings: { recursive: true },
  });
  const pickerState = await e2eState(request);
  expect(pickerState.picker_calls.at(-1)).toEqual({ kind: "folder", recursive: true });
  expect(pickerState.folder_paths[1]).toContain("Вложенная папка");

  await page.getByRole("button", { name: "Обновить" }).click();
  await expect.poll(() => requestPayloads.refresh.length).toBe(1);
  await expect(page.locator(".file-card")).toHaveCount(2);
  expect(requestPayloads.refresh[0].settings.recursive).toBe(true);

  const refreshCount = requestPayloads.refresh.length;
  await page.getByLabel("Базовая длина строки SRT").fill("39");
  await expect
    .poll(() =>
      page.evaluate(
        (key) => JSON.parse(localStorage.getItem(key) ?? "{}").max_chars_per_line,
        SETTINGS_KEY,
      ),
    )
    .toBe(39);
  expect(requestPayloads.refresh).toHaveLength(refreshCount);

  let transcribePayload = null;
  await page.route("**/api/transcribe", async (route) => {
    transcribePayload = route.request().postDataJSON();
    await route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Контрольная ошибка постановки E2E." }),
    });
  });
  await page.getByRole("button", { name: "Запустить" }).click();
  await expect.poll(() => transcribePayload).not.toBeNull();
  expect(transcribePayload.settings.recursive).toBe(true);
  await expect(page.locator("#logConsole")).toContainText("Контрольная ошибка постановки E2E");

  const stored = await page.evaluate(
    (key) => JSON.parse(localStorage.getItem(key) ?? "{}"),
    SETTINGS_KEY,
  );
  expect(stored).not.toHaveProperty("recursive");
});

test("потеря состояния подготовки остаётся видимой", async ({ page, request }) => {
  let statusRequests = 0;
  await page.route("**/api/preparation-status", async (route) => {
    statusRequests += 1;
    if (statusRequests === 1) {
      await route.continue();
      return;
    }
    await route.abort("connectionfailed");
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Папка" }).click();
  try {
    await expect
      .poll(async () => (await e2eState(request)).folder_preparation_entered)
      .toBe(true);
    await expect(page.locator("#logConsole")).toContainText(
      "Потеряна связь с локальным сервисом во время подготовки",
    );
    await expect(page.locator("#transcribeBtn")).toBeDisabled();
    const requestsBeforeRepeat = statusRequests;
    await expect
      .poll(() => statusRequests)
      .toBeGreaterThanOrEqual(requestsBeforeRepeat + 2);
    await expect
      .poll(() => logTextCount(page, "Потеряна связь с локальным сервисом во время подготовки"))
      .toBe(1);
  } finally {
    await releaseFolderPreparation(request);
  }
  await expect(page.locator(".file-card")).toHaveCount(2);
});

test("необработанные ошибки браузера отображаются в журнале", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => {
    window.dispatchEvent(new ErrorEvent("error", {
      message: "Контрольная ошибка интерфейса E2E",
      error: new Error("Контрольная ошибка интерфейса E2E"),
    }));
    const rejection = new Event("unhandledrejection");
    Object.defineProperty(rejection, "reason", {
      value: new Error("Контрольное отклонение операции E2E"),
    });
    window.dispatchEvent(rejection);
  });

  await expect(page.locator("#logConsole")).toContainText(
    "Необработанная ошибка интерфейса: Контрольная ошибка интерфейса E2E",
  );
  await expect(page.locator("#logConsole")).toContainText(
    "Необработанная ошибка операции: Контрольное отклонение операции E2E",
  );
  await expect(page.locator("#terminalStatus")).toContainText("ошибка");
});

test("список из 10 000 файлов отображается страницами без блокировки интерфейса", async ({ page }) => {
  const items = Array.from({ length: LARGE_ITEM_COUNT }, (_, index) => {
    const number = String(index + 1).padStart(5, "0");
    const path = `D:\\E2E\\Большой пакет\\Запись ${number}.mkv`;
    return {
      id: `large-${number}`,
      path,
      name: `Запись ${number}.mkv`,
      state: "queued",
      stage: "В очереди",
      progress: 0,
      error: null,
    };
  });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.route("**/api/pick", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cancelled: false,
        mode: "files",
        paths: items.map((item) => item.path),
        items,
      }),
    });
  });

  await page.goto("/");
  await page.evaluate(() => {
    const probe = { last: performance.now(), maxGap: 0, ticks: 0 };
    window.__largeRenderProbe = probe;
    window.__largeRenderTimer = window.setInterval(() => {
      const now = performance.now();
      probe.maxGap = Math.max(probe.maxGap, now - probe.last);
      probe.last = now;
      probe.ticks += 1;
    }, 10);
  });

  const startedAt = Date.now();
  await page.getByRole("button", { name: "Файлы" }).click();
  await expect(page.locator("#statusSummary")).toContainText("10000 файлов", {
    timeout: 10_000,
  });
  await expect(page.locator("#fileList")).toHaveAttribute("data-total-count", "10000");
  await expect(page.locator("#fileList")).toHaveAttribute("data-rendered-count", "200");
  await expect(page.locator(".file-card")).toHaveCount(200);
  await expect(page.locator(".file-pagination-status")).toHaveText(
    "1/50 · карточки 1–200 из 10000",
  );
  await expect(page.locator(".file-name").first()).toHaveText("Запись 00001.mkv");

  await page.getByRole("button", { name: "Последняя страница" }).click();
  await expect(page.locator(".file-card")).toHaveCount(200);
  await expect(page.locator(".file-pagination-status")).toHaveText(
    "50/50 · карточки 9801–10000 из 10000",
  );
  await expect(page.locator(".file-name").last()).toHaveText("Запись 10000.mkv");

  const result = await page.evaluate(() => {
    clearInterval(window.__largeRenderTimer);
    return {
      probe: window.__largeRenderProbe,
    };
  });
  expect(Date.now() - startedAt).toBeLessThan(10_000);
  expect(result.probe.ticks).toBeGreaterThan(10);
  expect(result.probe.maxGap).toBeLessThan(1000);
  expect(pageErrors).toEqual([]);
});

test("длинная сводка пакета остаётся внутри шапки файлов", async ({ page }) => {
  const states = [
    ...Array(57).fill("queued"),
    "transcribing",
    ...Array(70).fill("done"),
    ...Array(5).fill("cached"),
    ...Array(4).fill("skipped"),
    ...Array(5).fill("error"),
    ...Array(3).fill("cancelled"),
    ...Array(3).fill("interrupted"),
  ];
  const items = states.map((state, index) => {
    const number = String(index + 1).padStart(3, "0");
    return {
      id: `summary-${number}`,
      path: `D:\\E2E\\Очень длинный путь к учебному курсу с вложенными каталогами\\Запись ${number}.mkv`,
      name: `Запись ${number}.mkv`,
      state,
      stage: "Контроль геометрии",
      progress: state === "done" || state === "cached" ? 100 : 0,
      error: state === "error" ? "Контрольная ошибка" : null,
    };
  });
  const expectedSummary = [
    "148 файлов",
    "в очереди: 57",
    "в работе: 1",
    "готово: 75",
    "пропущено: 4",
    "ошибок: 5",
    "отменено: 6",
  ].join(" · ");
  await page.route("**/api/pick", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cancelled: false,
        mode: "folder",
        path: "D:\\E2E\\Очень длинный путь к учебному курсу с вложенными каталогами",
        items,
      }),
    });
  });

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await page.getByRole("button", { name: "Папка" }).click();
  await expect(page.locator("#statusSummary")).toHaveText(expectedSummary);

  for (const width of [1440, 1120, 1040, 901, 900, 680, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const geometry = await page.evaluate(() => {
      const panel = document.querySelector(".panel-files");
      const header = panel.querySelector(".panel-header");
      const lead = header.firstElementChild;
      const summary = document.querySelector("#statusSummary");
      const headerRect = header.getBoundingClientRect();
      const leadRect = lead.getBoundingClientRect();
      const summaryRect = summary.getBoundingClientRect();
      const overlapX = Math.max(
        0,
        Math.min(leadRect.right, summaryRect.right) - Math.max(leadRect.left, summaryRect.left),
      );
      const overlapY = Math.max(
        0,
        Math.min(leadRect.bottom, summaryRect.bottom) - Math.max(leadRect.top, summaryRect.top),
      );
      return {
        headerFits: header.scrollWidth <= header.clientWidth + 1,
        panelFits: panel.scrollWidth <= panel.clientWidth + 1,
        summaryFits: summary.scrollWidth <= summary.clientWidth + 1,
        summaryInside:
          summaryRect.left >= headerRect.left - 1 &&
          summaryRect.right <= headerRect.right + 1 &&
          summaryRect.top >= headerRect.top - 1 &&
          summaryRect.bottom <= headerRect.bottom + 1,
        blocksDoNotOverlap: overlapX * overlapY === 0,
        pageFits: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
      };
    });
    expect(geometry, `ширина окна ${width}px`).toEqual({
      headerFits: true,
      panelFits: true,
      summaryFits: true,
      summaryInside: true,
      blocksDoNotOverlap: true,
      pageFits: true,
    });
  }
});

test("карточка показывает только корректную длительность медиа", async ({ page }) => {
  const cases = [
    ["Нет данных.mp4"],
    ["Null.mp4", null],
    ["Ноль.mp4", 0],
    ["Минус.mp4", -1],
    ["Строка.mp4", "3600"],
    ["Секунды.mp4", 59],
    ["Минута.mp4", 60],
    ["Час.mp4", 3600],
    ["Час и минута.mp4", 3661],
  ];
  const items = cases.map(([name, duration], index) => ({
    id: `duration-${index}`,
    path: `D:\\E2E\\${name}`,
    name,
    state: "done",
    stage: "Готово",
    progress: 100,
    ...(index === 0
      ? {}
      : { probe: { duration, format_name: "mp4", streams: [] } }),
  }));
  await page.route("**/api/pick", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        cancelled: false,
        mode: "files",
        paths: items.map((item) => item.path),
        items,
      }),
    });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Файлы" }).click();
  for (const name of cases.slice(0, 5).map(([name]) => name)) {
    await expect(fileDuration(page, name)).toHaveCount(0);
  }
  await expect(fileDuration(page, "Секунды.mp4")).toHaveText("0:59");
  await expect(fileDuration(page, "Минута.mp4")).toHaveText("1:00");
  await expect(fileDuration(page, "Час.mp4")).toHaveText("1:00:00");
  await expect(fileDuration(page, "Час и минута.mp4")).toHaveText("1:01:01");
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
  await expect(fileDuration(page, "Лекция 01.mp4")).toHaveText("1:00");
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveText("1:02");

  await page.locator("#advancedSettings > summary").click();
  const force = page.getByLabel("Перезаписать целевые результаты");
  const buildsBeforeForce = (await e2eState(request)).build_settings.length;
  await force.check();
  await expect(force).toBeChecked();
  await expect(page.locator('.file-card[data-status="cached"]')).toHaveCount(0);
  const storedBeforeStart = await page.evaluate(
    (key) => JSON.parse(localStorage.getItem(key) ?? "{}"),
    SETTINGS_KEY,
  );
  expect(storedBeforeStart).not.toHaveProperty("force");
  expect((await e2eState(request)).build_settings).toHaveLength(buildsBeforeForce);

  await page.getByRole("button", { name: "Запустить" }).click();
  await expect(force).not.toBeChecked();
  await expect
    .poll(async () => (await e2eState(request)).process_settings[0]?.force)
    .toBe(true);
  await expect(page.locator("#logConsole")).toContainText(
    "Начата постановка выбранных источников в очередь.",
  );
  const preparationStatus = await request.get("/api/preparation-status");
  expect(preparationStatus.ok()).toBeTruthy();
  expect(await preparationStatus.json()).toMatchObject({
    operation: "transcribe",
    status: "done",
    phase: "done",
    active: false,
  });
  await expect(fileCard(page, "Лекция 01.mp4")).toHaveAttribute("data-status", "done");
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toHaveAttribute(
    "data-status",
    "transcribing",
  );
  await expect(fileCard(page, "Разбор алгоритма №2.mkv")).toContainText("42%");
  await expect(fileDuration(page, "Лекция 01.mp4")).toHaveText("1:00");
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveCount(0);
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
  await expect(fileDuration(page, "Лекция 01.mp4")).toHaveText("1:00");
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveCount(0);
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
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveCount(0);
  await expect(retry).toBeEnabled();
  await expect(language).toBeEnabled();
  await expect(device).toBeEnabled();
  await expect.poll(() => controlLineCount(page)).toBe(1);

  const restarted = await request.post("/__e2e__/restart", {
    headers: LOCAL_ORIGIN_HEADERS,
  });
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
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
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
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveText("1:02");
  await expect.poll(async () => (await e2eState(request)).retry_entered).toBe(true);
  expect(await e2eState(request)).toMatchObject({ calls: [PATHS, PATHS.slice(1)] });
  await expect
    .poll(async () => (await e2eState(request)).process_settings[1]?.force)
    .toBe(false);

  const release = await request.post("/__e2e__/release", {
    headers: LOCAL_ORIGIN_HEADERS,
  });
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
  await expect(fileDuration(page, "Разбор алгоритма №2.mkv")).toHaveText("1:01");
  await expect(fileDuration(page, "Ошибка дорожки 03.wav")).toHaveText("1:02");
  await expect(retry).toBeEnabled();
  await expect.poll(() => controlLineCount(page)).toBe(1);

  const history = await request.get("/api/jobs?limit=2");
  expect(history.ok()).toBeTruthy();
  const jobs = (await history.json()).jobs;
  expect(jobs).toHaveLength(2);
  expect(jobs[0].retry_of).toBe(originalJobId);
  expect(jobs[1].job_id).toBe(originalJobId);
});
