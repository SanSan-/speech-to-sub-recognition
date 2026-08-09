"use strict";

const picker = document.getElementById("picker");
const pickerHint = document.getElementById("pickerHint");
const pickerFileBtn = document.getElementById("pickerFileBtn");
const pickerFolderBtn = document.getElementById("pickerFolderBtn");
const refreshBtn = document.getElementById("refreshBtn");
const unloadBtn = document.getElementById("unloadBtn");
const cancelBtn = document.getElementById("cancelBtn");
const retryBtn = document.getElementById("retryBtn");
const transcribeBtn = document.getElementById("transcribeBtn");
const fileList = document.getElementById("fileList");
const selectedPath = document.getElementById("selectedPath");
const statusSummary = document.getElementById("statusSummary");
const settingsForm = document.getElementById("settingsForm");
const backendSelect = document.getElementById("backend");
const modelPathInput = document.getElementById("modelPath");
const cloudSettings = document.getElementById("cloudSettings");
const allowCloudProcessingInput = document.getElementById("allowCloudProcessing");
const forceInput = document.getElementById("force");
const openAiAvailability = document.getElementById("openAiAvailability");
const alignerSelect = document.getElementById("aligner");
const alignerModelPathInput = document.getElementById("alignerModelPath");
const deviceSelect = document.getElementById("device");
const languageSelect = document.getElementById("language");
const logConsole = document.getElementById("logConsole");
const terminalStatus = document.getElementById("terminalStatus");
const clearLogs = document.getElementById("clearLogs");
const brandSubtitle = document.getElementById("brandSubtitle");
const settingsModeHint = document.getElementById("settingsModeHint");
const localAsrControls = Array.from(document.querySelectorAll("[data-local-asr-only]"));

const SETTINGS_STORAGE_KEY = "speechToSubSettingsV1";
const BROWSER_LOG_LIMIT = 500;
const ACTIVE_STATES = new Set([
  "queued",
  "probing",
  "extracting",
  "downloading",
  "transcribing",
  "aligning",
  "writing",
]);
const CANCELLED_STATES = new Set(["cancelled", "interrupted"]);
const RETRYABLE_STATES = new Set(["error", ...CANCELLED_STATES]);
const FINAL_STATES = new Set(["cached", "skipped", "done", "error", ...CANCELLED_STATES]);
const JOB_ID_PATTERN = /^[0-9a-f]{32}$/;
const STREAM_PATH_PREFIX = "/api/stream/";
const STREAM_ALLOWED_ORIGINS = Object.freeze([window.location.origin]);

const BUILTIN_DEFAULTS = {
  backend: "faster-whisper",
  model_path: "resources/models/whisper-large-v3-ct2",
  aligner: "none",
  aligner_model_path: null,
  worker_python_path: null,
  aligner_worker_python_path: null,
  language: "en",
  device: "auto",
  audio_language: "eng",
  audio_stream_index: null,
  quantization_enabled: true,
  auto_download_model: true,
  allow_cpu_fallback: false,
  allow_cloud_processing: false,
  openai_model: "whisper-1",
  max_chars_per_line: 42,
  line_length_gap: 8,
  max_cps: 17,
  long_form_window_seconds: 300,
  long_form_overlap_seconds: 2,
  vad_filter: true,
  vad_min_silence_ms: 600,
  beam_size: 5,
  condition_on_previous_text: true,
  keep_audio: false,
  force: false,
  recursive: false,
};

const statusLabels = {
  queued: "в очереди",
  probing: "проверка медиа",
  extracting: "извлечение аудио",
  downloading: "загрузка модели",
  transcribing: "распознавание",
  aligning: "выравнивание слов",
  writing: "запись результата",
  cached: "кеш",
  skipped: "пропущено",
  done: "готово",
  error: "ошибка",
  cancelled: "отменено",
  interrupted: "прервано",
};

const defaultProgress = {
  queued: 0,
  probing: 5,
  extracting: 15,
  downloading: 18,
  transcribing: 25,
  aligning: 86,
  writing: 92,
  cached: 100,
  skipped: 100,
  done: 100,
  error: 100,
  cancelled: 100,
  interrupted: 100,
};

const terminalLabels = {
  idle: "ожидание",
  running: "в работе",
  ok: "успешно",
  partial: "частично",
  error: "ошибка",
  cancelled: "отменено",
  interrupted: "прервано",
};

const state = {
  items: [],
  itemMap: new Map(),
  mode: /** @type {string | null} */ (null),
  sourcePaths: [],
  selectionLabel: "Ничего не выбрано",
  jobId: null,
  displayedJobId: null,
  eventCursor: 0,
  jobTotal: 0,
  jobDone: 0,
  completed: new Set(),
  isBusy: false,
  eventSource: /** @type {EventSource | null} */ (null),
  uiConfig: null,
  itemRequestGeneration: 0,
  itemRequestController: null,
  canRetry: false,
  cancelRequested: false,
  logLines: [],
};

function isCloudBackend() {
  return backendSelect.value === "openai-api";
}

function updatePickerHint() {
  if (!pickerHint) {
    return;
  }
  const extensions = Array.isArray(state.uiConfig?.supported_extensions)
    ? state.uiConfig.supported_extensions.map((extension) => String(extension).toUpperCase()).join(", ")
    : "";
  if (isCloudBackend()) {
    pickerHint.textContent = extensions
      ? `Медиа для OpenAI: ${extensions}`
      : "Выбранные медиа будут переданы в OpenAI при запуске";
    pickerHint.title = "Передача начинается только после отдельного явного согласия.";
    return;
  }
  pickerHint.textContent = extensions ? `Локальные файлы: ${extensions}` : "Файлы остаются на компьютере";
  pickerHint.title = "Файлы остаются на компьютере и не загружаются в браузер.";
}

function updateBackendMode(options = {}) {
  const cloud = isCloudBackend();
  if (options.resetConsent && allowCloudProcessingInput) {
    allowCloudProcessingInput.checked = false;
  }
  if (cloudSettings) {
    cloudSettings.hidden = !cloud;
    cloudSettings.querySelectorAll("[data-setting]").forEach((input) => {
      input.disabled = state.isBusy || !cloud;
    });
  }
  localAsrControls.forEach((control) => {
    control.hidden = cloud;
    control.querySelectorAll("[data-setting]").forEach((input) => {
      input.disabled = state.isBusy || cloud;
    });
  });
  brandSubtitle.textContent = cloud
    ? "Облачное пакетное распознавание речи в SRT"
    : "Локальное пакетное распознавание речи в SRT";
  settingsModeHint.textContent = cloud
    ? "Аудио передаётся в OpenAI только после явного согласия"
    : "Сеть при распознавании не используется";
  unloadBtn.textContent = cloud ? "Сбросить подключение" : "Выгрузить";
  unloadBtn.title = cloud
    ? "Закрыть клиент OpenAI и удалить ключ из памяти процесса"
    : "Выгрузить локальную модель из памяти";
  if (openAiAvailability) {
    openAiAvailability.textContent = state.uiConfig?.openai_configured
      ? "Ключ API настроен в окружении."
      : "Ключ API не найден в окружении.";
  }
  updatePickerHint();
  const hasSelection = state.items.length > 0 || state.sourcePaths.length > 0;
  const cloudAllowed = !cloud || Boolean(allowCloudProcessingInput?.checked);
  transcribeBtn.disabled = state.isBusy || !hasSelection || !cloudAllowed;
  retryBtn.disabled = state.isBusy || !state.canRetry || !state.displayedJobId || !cloudAllowed;
}

function appendLog(message) {
  if (message === undefined || message === null) {
    return;
  }
  const lines = String(message).replaceAll("\r\n", "\n").split("\n");
  if (lines.at(-1) === "") {
    lines.pop();
  }
  state.logLines.push(...lines);
  if (state.logLines.length > BROWSER_LOG_LIMIT) {
    state.logLines.splice(0, state.logLines.length - BROWSER_LOG_LIMIT);
  }
  logConsole.textContent = state.logLines.length > 0 ? `${state.logLines.join("\n")}\n` : "";
  logConsole.scrollTop = logConsole.scrollHeight;
}

function setTerminalState(status) {
  const normalized = Object.hasOwn(terminalLabels, status) ? status : "idle";
  terminalStatus.className = `terminal-status ${normalized}`;
  terminalStatus.textContent = terminalLabels[normalized];
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  if (isBusy) {
    abortItemRequest();
  }
  const hasItems = state.items.length > 0;
  const hasSelection = hasItems || state.sourcePaths.length > 0;
  transcribeBtn.disabled = isBusy || !hasSelection;
  refreshBtn.disabled = isBusy || !hasSelection;
  unloadBtn.disabled = isBusy;
  cancelBtn.disabled = !isBusy || !state.jobId || state.cancelRequested;
  retryBtn.disabled = isBusy || !state.canRetry || !state.displayedJobId;
  picker.classList.toggle("disabled", isBusy);
  picker.setAttribute("aria-disabled", String(isBusy));
  pickerFileBtn.disabled = isBusy;
  pickerFolderBtn.disabled = isBusy;
  settingsForm.querySelectorAll("[data-setting]").forEach((input) => {
    input.disabled = isBusy;
  });
  updateBackendMode();
  if (!isBusy) {
    transcribeBtn.textContent = "Запустить";
  }
}

function abortItemRequest() {
  state.itemRequestGeneration += 1;
  if (state.itemRequestController) {
    state.itemRequestController.abort();
    state.itemRequestController = null;
  }
}

function beginItemRequest() {
  abortItemRequest();
  const controller = new AbortController();
  state.itemRequestController = controller;
  return { controller, generation: state.itemRequestGeneration };
}

function completeItemRequest(controller) {
  if (state.itemRequestController === controller) {
    state.itemRequestController = null;
  }
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function updateJobProgress() {
  if (!state.isBusy) {
    transcribeBtn.textContent = "Запустить";
    return;
  }
  if (state.jobTotal > 0) {
    transcribeBtn.textContent = `Обработка ${state.jobDone}/${state.jobTotal}`;
    return;
  }
  transcribeBtn.textContent = "Запуск…";
}

function debounce(callback, delay) {
  let timer = null;
  return (...args) => {
    if (timer !== null) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => callback(...args), delay);
  };
}

function clampProgress(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round(parsed)));
}

function asFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function setSelectOptions(select, options, fallbackValue) {
  if (!select || !Array.isArray(options) || options.length === 0) {
    return;
  }
  const currentValue = select.value;
  const fragment = document.createDocumentFragment();
  const values = [];
  options.forEach((item) => {
    const value = typeof item === "object" && item !== null ? item.value : item;
    if (value === undefined || value === null) {
      return;
    }
    const normalized = String(value);
    const label = typeof item === "object" && item !== null && item.label !== undefined
      ? String(item.label)
      : normalized;
    const option = document.createElement("option");
    option.value = normalized;
    option.textContent = label;
    fragment.append(option);
    values.push(normalized);
  });
  if (values.length === 0) {
    return;
  }
  select.innerHTML = "";
  select.append(fragment);
  const fallback = fallbackValue ?? "";
  const preferred = values.includes(currentValue) ? currentValue : String(fallback);
  select.value = values.includes(preferred) ? preferred : values[0];
  select.defaultValue = select.value;
}

function setInputValue(input, value) {
  if (!input) {
    return;
  }
  if (input.type === "checkbox") {
    const checked = Boolean(value);
    input.checked = checked;
    input.defaultChecked = checked;
    return;
  }
  const normalized = value === undefined || value === null ? "" : String(value);
  if (input instanceof HTMLSelectElement) {
    const exists = Array.from(input.options).some((option) => option.value === normalized);
    if (!exists) {
      return;
    }
  }
  input.value = normalized;
  input.defaultValue = normalized;
}

function applySettingValues(values) {
  if (!values || typeof values !== "object") {
    return;
  }
  settingsForm.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (!["allow_cloud_processing", "force"].includes(key) && Object.hasOwn(values, key)) {
      setInputValue(input, values[key]);
    }
  });
  if (allowCloudProcessingInput) {
    allowCloudProcessingInput.checked = false;
  }
  if (forceInput) {
    forceInput.checked = false;
  }
}

function loadStoredSettings() {
  try {
    const raw = window.localStorage ? localStorage.getItem(SETTINGS_STORAGE_KEY) : null;
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    delete parsed.allow_cloud_processing;
    delete parsed.force;
    return parsed;
  } catch {
    appendLog("Не удалось прочитать сохранённые настройки.");
    return null;
  }
}

function readSettings() {
  const settings = {};
  settingsForm.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (input.type === "checkbox") {
      settings[key] = input.checked;
      return;
    }
    if (input.type === "number") {
      const value = input.value.trim();
      settings[key] = value === "" ? null : Number(value);
      return;
    }
    const value = input.value.trim();
    settings[key] = value === "" ? input.defaultValue.trim() : value;
  });
  return settings;
}

function persistSettings() {
  try {
    if (window.localStorage) {
      const settings = readSettings();
      delete settings.allow_cloud_processing;
      delete settings.force;
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    }
  } catch (error) {
    // Ошибка хранилища не должна блокировать локальное распознавание.
  }
}

function consumeOneShotForce() {
  const allowed = Boolean(forceInput?.checked);
  if (forceInput) {
    forceInput.checked = false;
  }
  persistSettings();
  return allowed;
}

function applyBackendModelDefault() {
  if (!state.uiConfig || !Array.isArray(state.uiConfig.backends) || !modelPathInput) {
    return;
  }
  const selected = state.uiConfig.backends.find((item) => item.value === backendSelect.value);
  if (!selected?.model_path) {
    return;
  }
  const knownDefaults = state.uiConfig.backends
    .map((item) => String(item.model_path || ""))
    .filter(Boolean);
  const current = modelPathInput.value.trim();
  if (!current || knownDefaults.includes(current)) {
    setInputValue(modelPathInput, selected.model_path);
  }
}

function selectedAlignerConfig() {
  if (!state.uiConfig || !Array.isArray(state.uiConfig.aligners) || !alignerSelect) {
    return null;
  }
  return state.uiConfig.aligners.find((item) => item.value === alignerSelect.value) || null;
}

function applyAlignerModelDefault() {
  if (!alignerModelPathInput) {
    return;
  }
  const selected = selectedAlignerConfig();
  if (!selected) {
    return;
  }
  const knownDefaults = state.uiConfig.aligners
    .map((item) => String(item.model_path || ""))
    .filter(Boolean);
  const current = alignerModelPathInput.value.trim();
  if (!current || knownDefaults.includes(current)) {
    setInputValue(alignerModelPathInput, selected.model_path || "");
  }
}

function compatibleBackendsForSelectedAligner() {
  const selected = selectedAlignerConfig();
  return selected && Array.isArray(selected.compatible_backends)
    ? selected.compatible_backends
    : [];
}

function enforceAlignerCompatibility() {
  const compatible = compatibleBackendsForSelectedAligner();
  if (compatible.length > 0 && !compatible.includes(backendSelect.value)) {
    setInputValue(alignerSelect, "none");
  }
  applyAlignerModelDefault();
}

function enforceBackendCompatibilityForAligner() {
  const compatible = compatibleBackendsForSelectedAligner();
  if (compatible.length > 0 && !compatible.includes(backendSelect.value)) {
    setInputValue(backendSelect, compatible[0]);
    applyBackendModelDefault();
  }
  applyAlignerModelDefault();
}

function formatDetail(detail) {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg ? item.msg : String(item)).join("; ");
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch (error) {
      return "Неизвестная ошибка API";
    }
  }
  return "Неизвестная ошибка API";
}

async function getJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    signal: options.signal,
  });
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function postJson(url, payload, options = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: options.signal,
  });

  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data?.detail !== undefined) {
        detail = formatDetail(data.detail);
      }
    } catch {
      appendLog("Ответ API с ошибкой не содержит корректный JSON.");
    }
    const requestError = new Error(detail);
    requestError.status = response.status;
    throw requestError;
  }

  if (response.status === 204) {
    return {};
  }
  return response.json();
}

function fileExtension(item) {
  if (item.format) {
    return String(item.format).replace(/^\./, "").toUpperCase();
  }
  const source = String(item.name || item.path || "");
  const match = /\.([^./\\]+)$/.exec(source);
  return match ? match[1].toUpperCase() : "MEDIA";
}

function fileName(item) {
  if (item.name) {
    return String(item.name);
  }
  const parts = String(item.path || "").split(/[\\/]/);
  return parts.at(-1) || "Без имени";
}

function normalizeState(item) {
  if (item?.state && statusLabels[item.state]) {
    return item.state;
  }
  if (item?.cached) {
    return "cached";
  }
  if (item?.skipped) {
    return "skipped";
  }
  return "queued";
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) {
    return null;
  }
  const rounded = Math.round(value);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const rest = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function appendMeta(meta, text, className) {
  if (!text) {
    return null;
  }
  const element = document.createElement("span");
  if (className) {
    element.className = className;
  }
  element.textContent = text;
  meta.append(element);
  return element;
}

function updateMeta(entry, item) {
  entry.metaEl.innerHTML = "";
  appendMeta(entry.metaEl, fileExtension(item), "badge");
  const probe = item?.probe && typeof item.probe === "object" ? item.probe : null;
  const duration = probe ? formatDuration(probe.duration) : null;
  if (duration) {
    appendMeta(entry.metaEl, duration);
  }
  if (probe && Array.isArray(probe.streams)) {
    appendMeta(entry.metaEl, `аудиопотоков: ${probe.streams.length}`);
  }
  if (item.stage) {
    entry.stageEl = appendMeta(entry.metaEl, String(item.stage), "stage-badge");
  } else {
    entry.stageEl = null;
  }
}

function collectNotice(item) {
  if (item.error) {
    return {
      kind: CANCELLED_STATES.has(normalizeState(item)) ? "warning" : "error",
      text: String(item.error),
    };
  }
  const warnings = [];
  if (item.warning) {
    warnings.push(String(item.warning));
  }
  if (Array.isArray(item.warnings)) {
    item.warnings.forEach((warning) => warnings.push(String(warning)));
  }
  if (item.probe?.warning) {
    warnings.push(String(item.probe.warning));
  }
  if (item.probe && Array.isArray(item.probe.warnings)) {
    item.probe.warnings.forEach((warning) => warnings.push(String(warning)));
  }
  if (warnings.length > 0) {
    return { kind: "warning", text: warnings.join(" ") };
  }
  if (normalizeState(item) === "skipped") {
    return { kind: "warning", text: "Существующий результат оставлен без изменений." };
  }
  return null;
}

function updateOutputs(entry, item) {
  const outputs = [
    ["SRT", item.srt_output],
    ["JSON", item.sidecar_output],
    ["FLAC", item.audio_output],
  ].filter(([, path]) => Boolean(path));

  entry.outputsEl.innerHTML = "";
  entry.outputsEl.hidden = outputs.length === 0;
  outputs.forEach(([label, path]) => {
    const row = document.createElement("div");
    row.className = "file-output-item";

    const labelEl = document.createElement("span");
    labelEl.className = "output-label";
    labelEl.textContent = label;

    const pathEl = document.createElement("span");
    pathEl.className = "output-path";
    pathEl.textContent = String(path);
    pathEl.title = String(path);

    row.append(labelEl, pathEl);
    entry.outputsEl.append(row);
  });
}

function updateNotice(entry, item) {
  const notice = collectNotice(item);
  entry.noticeEl.hidden = notice === null;
  entry.noticeEl.className = notice?.kind === "error" ? "file-error" : "file-warning";
  entry.noticeEl.textContent = notice ? notice.text : "";
  entry.card.title = item.error ? String(item.error) : "";
}

function progressFor(item, status) {
  const explicit = clampProgress(item.progress);
  return explicit === null ? defaultProgress[status] || 0 : explicit;
}

function buildFileCard(item, index) {
  const status = normalizeState(item);
  const card = document.createElement("article");
  card.className = "file-card";
  card.dataset.path = String(item.path || "");
  card.dataset.status = status;
  card.style.setProperty("--delay", `${Math.min(index, 6) * 0.045}s`);

  const row = document.createElement("div");
  row.className = "file-row";

  const dot = document.createElement("span");
  dot.className = "file-dot";
  dot.setAttribute("aria-hidden", "true");

  const info = document.createElement("div");
  info.className = "file-info";

  const name = document.createElement("div");
  name.className = "file-name";
  name.textContent = fileName(item);
  name.title = fileName(item);

  const path = document.createElement("div");
  path.className = "file-path";
  path.textContent = String(item.path || "");
  path.title = String(item.path || "");

  const meta = document.createElement("div");
  meta.className = "file-meta";
  info.append(name, path, meta);

  const statusEl = document.createElement("span");
  statusEl.className = "file-status";
  statusEl.textContent = statusLabels[status];

  row.append(dot, info, statusEl);

  const progress = document.createElement("div");
  progress.className = "file-progress";
  progress.setAttribute("role", "progressbar");
  progress.setAttribute("aria-label", `Прогресс: ${fileName(item)}`);
  progress.setAttribute("aria-valuemin", "0");
  progress.setAttribute("aria-valuemax", "100");

  const bar = document.createElement("span");
  bar.className = "file-progress-bar";
  progress.append(bar);

  const outputs = document.createElement("div");
  outputs.className = "file-output-list";

  const notice = document.createElement("div");
  notice.hidden = true;

  card.append(row, progress, outputs, notice);

  const entry = {
    card,
    statusEl,
    progressEl: progress,
    bar,
    metaEl: meta,
    stageEl: null,
    outputsEl: outputs,
    noticeEl: notice,
  };
  updateMeta(entry, item);
  updateOutputs(entry, item);
  updateNotice(entry, item);
  const progressValue = progressFor(item, status);
  bar.style.width = `${progressValue}%`;
  progress.setAttribute("aria-valuenow", String(progressValue));
  if (ACTIVE_STATES.has(status) && progressValue > 0) {
    statusEl.textContent = `${statusLabels[status]} · ${progressValue}%`;
  }
  return entry;
}

function renderEmpty() {
  fileList.classList.add("empty");
  const empty = document.createElement("div");
  empty.className = "empty-state";

  const icon = document.createElement("span");
  icon.className = "empty-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = "♪";

  const text = document.createElement("span");
  text.textContent = "Выберите один или несколько файлов либо папку.";
  empty.append(icon, text);
  fileList.append(empty);
}

function updateSummary() {
  const total = state.items.length;
  if (total === 0) {
    statusSummary.textContent = "0 файлов";
    return;
  }
  const counts = state.items.reduce((result, item) => {
    const itemState = normalizeState(item);
    result[itemState] = (result[itemState] || 0) + 1;
    return result;
  }, {});
  const ready = (counts.done || 0) + (counts.cached || 0);
  const active = ["probing", "extracting", "downloading", "transcribing", "writing"]
    .reduce((sum, key) => sum + (counts[key] || 0), 0);
  const parts = [`${total} файлов`];
  if (counts.queued) {
    parts.push(`в очереди: ${counts.queued}`);
  }
  if (active > 0) {
    parts.push(`в работе: ${active}`);
  }
  if (ready > 0) {
    parts.push(`готово: ${ready}`);
  }
  if (counts.skipped) {
    parts.push(`пропущено: ${counts.skipped}`);
  }
  if (counts.error) {
    parts.push(`ошибок: ${counts.error}`);
  }
  const cancelled = (counts.cancelled || 0) + (counts.interrupted || 0);
  if (cancelled) {
    parts.push(`отменено: ${cancelled}`);
  }
  statusSummary.textContent = parts.join(" · ");
}

function renderItems(items, label) {
  state.items = Array.isArray(items) ? items.map((item) => ({ ...item })) : [];
  state.itemMap = new Map();
  fileList.innerHTML = "";
  if (label) {
    state.selectionLabel = String(label);
  }
  selectedPath.textContent = state.selectionLabel;
  selectedPath.title = state.selectionLabel;

  if (state.items.length === 0) {
    renderEmpty();
    updateSummary();
    setBusy(state.isBusy);
    return;
  }

  fileList.classList.remove("empty");
  const fragment = document.createDocumentFragment();
  state.items.forEach((item, index) => {
    const entry = buildFileCard(item, index);
    state.itemMap.set(String(item.path || ""), entry);
    fragment.append(entry.card);
  });
  fileList.append(fragment);
  updateSummary();
  setBusy(state.isBusy);
}

function updateFileState(path, payload) {
  const normalizedPath = String(path || "");
  const entry = state.itemMap.get(normalizedPath);
  if (!entry) {
    return;
  }
  const item = state.items.find((candidate) => String(candidate.path || "") === normalizedPath);
  if (!item) {
    return;
  }
  Object.assign(item, payload);
  if (payload.status && !payload.state) {
    item.state = payload.status;
  }
  const status = normalizeState(item);
  const progress = progressFor(item, status);
  entry.card.dataset.status = status;
  entry.statusEl.textContent = ACTIVE_STATES.has(status) && progress > 0
    ? `${statusLabels[status]} · ${progress}%`
    : statusLabels[status];
  entry.bar.style.width = `${progress}%`;
  entry.progressEl.setAttribute("aria-valuenow", String(progress));
  updateMeta(entry, item);
  updateOutputs(entry, item);
  updateNotice(entry, item);
  updateSummary();
}

function selectionLabel(result, kind) {
  if (result.path) {
    return String(result.path);
  }
  const count = Array.isArray(result.items) ? result.items.length : 0;
  return kind === "folder" ? `Папка · ${count} файлов` : `Выбрано файлов: ${count}`;
}

async function pick(kind) {
  if (state.isBusy) {
    return;
  }
  const request = beginItemRequest();
  let result;
  try {
    result = await postJson(
      "/api/pick",
      { kind, settings: readSettings() },
      { signal: request.controller.signal },
    );
  } finally {
    completeItemRequest(request.controller);
  }
  if (request.generation !== state.itemRequestGeneration || state.isBusy) {
    return;
  }
  if (result.cancelled) {
    appendLog("Выбор отменён.");
    return;
  }
  state.mode = result.mode || kind;
  state.displayedJobId = null;
  state.canRetry = false;
  state.cancelRequested = false;
  if (state.mode === "folder" && result.path) {
    state.sourcePaths = [String(result.path)];
  } else if (Array.isArray(result.paths)) {
    state.sourcePaths = result.paths.map(String);
  } else {
    state.sourcePaths = (result.items || [])
      .map((item) => String(item.path || ""))
      .filter(Boolean);
  }
  renderItems(result.items || [], selectionLabel(result, state.mode));
  setTerminalState("idle");
}

async function runPick(kind) {
  try {
    await pick(kind);
  } catch (error) {
    if (isAbortError(error)) {
      return;
    }
    appendLog(`Ошибка выбора: ${error.message}`);
  }
}

async function refreshItems(options = {}) {
  if ((state.items.length === 0 && state.sourcePaths.length === 0) || state.isBusy) {
    return;
  }
  const request = beginItemRequest();
  const paths = state.sourcePaths.length > 0
    ? [...state.sourcePaths]
    : state.items.map((item) => item.path);
  let result;
  try {
    result = await postJson(
      "/api/refresh",
      { paths, settings: readSettings() },
      { signal: request.controller.signal },
    );
  } finally {
    completeItemRequest(request.controller);
  }
  if (request.generation !== state.itemRequestGeneration || state.isBusy) {
    return;
  }
  renderItems(result.items || [], result.path || state.selectionLabel);
  if (!options.silent) {
    appendLog("Список файлов обновлён.");
  }
}

function restoreCounters(snapshot, items) {
  state.jobTotal = asFiniteNumber(snapshot.total, items.length);
  state.completed = new Set();
  items.forEach((item) => {
    if (FINAL_STATES.has(normalizeState(item))) {
      state.completed.add(String(item.path || ""));
    }
  });
  state.jobDone = asFiniteNumber(snapshot.done, state.completed.size);
}

async function loadActiveJob(options = {}) {
  let result;
  try {
    result = await getJson("/api/active-job");
  } catch (error) {
    if (!options.quiet) {
      appendLog(`Ошибка синхронизации задачи: ${error.message}`);
    }
    return false;
  }
  if (!result || (!result.active && !result.terminal)) {
    return false;
  }

  abortItemRequest();

  const items = Array.isArray(result.items) ? result.items : [];
  state.displayedJobId = result.job_id || null;
  state.canRetry = Boolean(result.terminal) && items.some(
    (item) => RETRYABLE_STATES.has(normalizeState(item)),
  );
  state.cancelRequested = Boolean(result.cancel_requested);
  state.sourcePaths = Array.isArray(result.source_paths) && result.source_paths.length > 0
    ? result.source_paths.map(String)
    : items.map((item) => String(item.path || "")).filter(Boolean);
  state.eventCursor = Math.max(0, asFiniteNumber(result.latest_event_id, 0));
  renderItems(items, `Задача · ${items.length} файлов`);
  restoreCounters(result, items);
  if (Array.isArray(result.logs) && !options.preserveLogs) {
    state.logLines = [];
    logConsole.textContent = "";
    result.logs.forEach((line) => appendLog(line));
  }

  if (result.settings && typeof result.settings === "object") {
    applySettingValues(result.settings);
    persistSettings();
    updateBackendMode({ resetConsent: true });
  }
  if (result.active) {
    state.jobId = result.job_id;
    setTerminalState("running");
    setBusy(true);
    updateJobProgress();
    listenJob(result.job_id, state.eventCursor);
    return true;
  }

  state.jobId = null;
  state.eventCursor = 0;
  setBusy(false);
  setTerminalState(result.status || "idle");
  return true;
}

function terminalMessage(payload) {
  const status = payload.status || "error";
  const done = asFiniteNumber(payload.done, state.jobDone);
  const cached = asFiniteNumber(payload.cached, 0);
  const skipped = asFiniteNumber(payload.skipped, 0);
  const failed = asFiniteNumber(payload.failed, 0);
  const cancelled = asFiniteNumber(payload.cancelled, 0);
  let prefix = "Распознавание завершено с ошибкой";
  if (status === "ok") {
    prefix = "Распознавание завершено";
  } else if (status === "partial") {
    prefix = "Распознавание завершено частично";
  } else if (status === "cancelled") {
    prefix = "Распознавание отменено";
  } else if (status === "interrupted") {
    prefix = "Распознавание прервано перезапуском";
  }
  return `${prefix}: обработано ${done}, кеш ${cached}, пропущено ${skipped}, ошибок ${failed}, отменено ${cancelled}.`;
}

function finishJob(jobId, payload) {
  if (String(state.jobId) !== String(jobId)) {
    return;
  }
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.jobDone = asFiniteNumber(payload.done, state.jobDone);
  state.jobId = null;
  state.eventCursor = 0;
  state.cancelRequested = false;
  state.canRetry = state.items.some((item) => RETRYABLE_STATES.has(normalizeState(item)));
  setBusy(false);
  setTerminalState(payload.status || "error");
  appendLog(terminalMessage(payload));
}

async function recoverStream(jobId) {
  if (String(state.jobId) !== String(jobId)) {
    return;
  }
  const restored = await loadActiveJob({ quiet: true, preserveLogs: true });
  if (!restored && String(state.jobId) === String(jobId)) {
    state.jobId = null;
    setBusy(false);
    setTerminalState("error");
    appendLog("Не удалось восстановить состояние задачи.");
  }
}

function listenJob(jobId, cursor = 0) {
  if (!jobId) {
    return;
  }
  const normalizedJobId = String(jobId);
  if (!JOB_ID_PATTERN.test(normalizedJobId)) {
    throw new Error("Сервер вернул некорректный идентификатор задачи.");
  }
  if (state.eventSource) {
    state.eventSource.close();
  }
  const normalizedCursor = Math.max(0, asFiniteNumber(cursor, 0));
  state.eventCursor = normalizedCursor;
  const streamUrl = new URL(STREAM_PATH_PREFIX, window.location.origin);
  streamUrl.pathname += encodeURIComponent(normalizedJobId);
  streamUrl.searchParams.set("cursor", String(normalizedCursor));
  if (!STREAM_ALLOWED_ORIGINS.includes(streamUrl.origin)) {
    throw new Error("Адрес потока задачи не входит в список разрешённых.");
  }
  const stream = new EventSource(streamUrl.toString());
  state.eventSource = stream;

  stream.onmessage = (event) => {
    if (!event.data || String(state.jobId) !== String(jobId)) {
      return;
    }
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      appendLog("Получено некорректное событие прогресса.");
      return;
    }
    const receivedEventId = Number.parseInt(event.lastEventId, 10);
    if (Number.isFinite(receivedEventId)) {
      state.eventCursor = Math.max(state.eventCursor, receivedEventId);
    }
    if (payload.type === "log") {
      appendLog(payload.message);
      return;
    }
    if (payload.type === "job") {
      state.jobTotal = asFiniteNumber(payload.total, state.jobTotal);
      state.jobDone = asFiniteNumber(payload.done, state.jobDone);
      if (payload.status === "cancelling") {
        state.cancelRequested = true;
        setBusy(true);
      }
      updateJobProgress();
      return;
    }
    if (payload.type === "file") {
      updateFileState(payload.path, payload);
      const fileState = payload.state || payload.status;
      const path = String(payload.path || "");
      if (FINAL_STATES.has(fileState) && !state.completed.has(path)) {
        state.completed.add(path);
        state.jobDone += 1;
      }
      if (Number.isFinite(Number(payload.done))) {
        state.jobDone = Number(payload.done);
      }
      updateJobProgress();
      return;
    }
    if (payload.type === "done") {
      finishJob(jobId, payload);
    }
  };

  stream.onerror = () => {
    if (String(state.jobId) !== String(jobId)) {
      stream.close();
      return;
    }
    stream.close();
    if (state.eventSource === stream) {
      state.eventSource = null;
    }
    appendLog("Поток событий прерван, выполняется синхронизация.");
    setTimeout(() => {
      recoverStream(jobId).catch((error) => {
        appendLog(`Ошибка восстановления: ${error.message}`);
      });
    }, 900);
  };
}

async function transcribe() {
  if ((state.items.length === 0 && state.sourcePaths.length === 0) || state.isBusy) {
    return;
  }
  if (isCloudBackend() && !allowCloudProcessingInput.checked) {
    appendLog("Перед запуском облачного распознавания подтвердите передачу аудио в OpenAI.");
    updateBackendMode();
    return;
  }
  const cloud = isCloudBackend();
  const requestSettings = readSettings();
  consumeOneShotForce();
  if (cloud) {
    allowCloudProcessingInput.checked = false;
  }
  state.jobTotal = state.items.length;
  state.jobDone = 0;
  state.eventCursor = 0;
  state.completed = new Set();
  state.displayedJobId = null;
  state.canRetry = false;
  state.cancelRequested = false;
  setTerminalState("running");
  setBusy(true);
  updateJobProgress();
  state.items.forEach((item) => {
    updateFileState(item.path, { state: "queued", progress: 0, error: null });
  });

  try {
    const paths = state.sourcePaths.length > 0
      ? [...state.sourcePaths]
      : state.items.map((item) => item.path);
    const result = await postJson("/api/transcribe", { paths, settings: requestSettings });
    state.jobId = result.job_id;
    state.displayedJobId = result.job_id;
    setBusy(true);
    if (Array.isArray(result.items)) {
      renderItems(result.items, state.selectionLabel);
      state.jobTotal = result.items.length;
    } else {
      state.jobTotal = asFiniteNumber(result.total, state.jobTotal);
    }
    appendLog(cloud ? "Запущено облачное распознавание." : "Запущено локальное распознавание.");
    updateJobProgress();
    listenJob(result.job_id, 0);
  } catch (error) {
    if (error.status === 409) {
      appendLog("Обнаружена уже запущенная задача, выполняется синхронизация.");
      const restored = await loadActiveJob({ preserveLogs: true });
      if (restored) {
        return;
      }
    }
    appendLog(`Ошибка запуска: ${error.message}`);
    state.jobId = null;
    state.eventCursor = 0;
    setBusy(false);
    setTerminalState("error");
  }
}

async function cancelJob() {
  if (!state.isBusy || !state.jobId || state.cancelRequested) {
    return;
  }
  const jobId = state.jobId;
  state.cancelRequested = true;
  setBusy(true);
  try {
    await postJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {});
    appendLog("Запрошена безопасная отмена пакетной задачи.");
  } catch (error) {
    state.cancelRequested = false;
    setBusy(true);
    appendLog(`Ошибка отмены: ${error.message}`);
  }
}

async function retryFailed() {
  if (state.isBusy || !state.canRetry || !state.displayedJobId) {
    return;
  }
  if (isCloudBackend() && !allowCloudProcessingInput.checked) {
    appendLog("Перед повторным облачным запуском заново подтвердите передачу аудио в OpenAI.");
    updateBackendMode();
    return;
  }
  const sourceJobId = state.displayedJobId;
  const cloud = isCloudBackend();
  const cloudAllowed = Boolean(allowCloudProcessingInput.checked);
  const forceAllowed = consumeOneShotForce();
  if (cloud) {
    allowCloudProcessingInput.checked = false;
  }
  state.canRetry = false;
  state.cancelRequested = false;
  setTerminalState("running");
  setBusy(true);
  try {
    const result = await postJson(
      `/api/jobs/${encodeURIComponent(sourceJobId)}/retry`,
      { allow_cloud_processing: cloudAllowed, force: forceAllowed },
    );
    state.jobId = result.job_id;
    state.displayedJobId = result.job_id;
    state.eventCursor = 0;
    state.completed = new Set();
    const retryItems = Array.isArray(result.items) ? result.items : [];
    state.sourcePaths = retryItems
      .map((item) => String(item.path || ""))
      .filter(Boolean);
    renderItems(retryItems, `Повторный запуск · ${retryItems.length} файлов`);
    state.jobTotal = state.items.length;
    state.jobDone = 0;
    setBusy(true);
    appendLog("Запущена повторная обработка неуспешных файлов.");
    updateJobProgress();
    listenJob(result.job_id, 0);
  } catch (error) {
    state.canRetry = state.items.some((item) => RETRYABLE_STATES.has(normalizeState(item)));
    setBusy(false);
    setTerminalState("error");
    appendLog(`Ошибка повторного запуска: ${error.message}`);
  }
}

async function unloadModel() {
  if (state.isBusy) {
    return;
  }
  try {
    const result = await postJson("/api/unload", {});
    appendLog(result.message || "Локальная модель выгружена из памяти.");
  } catch (error) {
    appendLog(`Ошибка выгрузки модели: ${error.message}`);
  }
}

async function loadUiConfig() {
  const config = await getJson("/api/ui-config");
  state.uiConfig = config;
  const defaults = { ...BUILTIN_DEFAULTS, ...config.defaults };
  setSelectOptions(backendSelect, config.backends, defaults.backend);
  setSelectOptions(alignerSelect, config.aligners, defaults.aligner);
  setSelectOptions(deviceSelect, config.devices, defaults.device);
  setSelectOptions(languageSelect, config.languages, defaults.language);
  applySettingValues(defaults);
  const stored = loadStoredSettings();
  if (stored) {
    applySettingValues(stored);
  }
  applyBackendModelDefault();
  enforceAlignerCompatibility();
  updateBackendMode({ resetConsent: true });
  if (config.settings_warning) {
    appendLog(`Предупреждение настроек: ${config.settings_warning}`);
  }
}

pickerFileBtn.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  runPick("file");
});

pickerFolderBtn.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  runPick("folder");
});

picker.addEventListener("dragover", (event) => {
  if (state.isBusy) {
    return;
  }
  event.preventDefault();
  picker.classList.add("drag-over");
});

picker.addEventListener("dragleave", () => {
  picker.classList.remove("drag-over");
});

picker.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  picker.classList.remove("drag-over");
  appendLog("Перетаскивание отключено: используйте локальный выбор файлов или папки.");
});

refreshBtn.addEventListener("click", () => {
  refreshItems().catch((error) => {
    if (!isAbortError(error)) {
      appendLog(`Ошибка обновления: ${error.message}`);
    }
  });
});

transcribeBtn.addEventListener("click", () => {
  transcribe().catch((error) => appendLog(`Ошибка запуска: ${error.message}`));
});

cancelBtn.addEventListener("click", () => {
  cancelJob().catch((error) => appendLog(`Ошибка отмены: ${error.message}`));
});

retryBtn.addEventListener("click", () => {
  retryFailed().catch((error) => appendLog(`Ошибка повторного запуска: ${error.message}`));
});

unloadBtn.addEventListener("click", () => {
  unloadModel().catch((error) => appendLog(`Ошибка выгрузки модели: ${error.message}`));
});

clearLogs.addEventListener("click", () => {
  state.logLines = [];
  logConsole.textContent = "";
});

backendSelect.addEventListener("change", () => {
  applyBackendModelDefault();
  enforceAlignerCompatibility();
  updateBackendMode({ resetConsent: true });
});
alignerSelect.addEventListener("change", enforceBackendCompatibilityForAligner);
allowCloudProcessingInput.addEventListener("change", () => {
  updateBackendMode();
});

const refreshDebounced = debounce(() => {
  refreshItems({ silent: true }).catch((error) => {
    if (!isAbortError(error)) {
      appendLog(`Ошибка обновления параметров: ${error.message}`);
    }
  });
}, 600);

settingsForm.addEventListener("input", () => {
  persistSettings();
  refreshDebounced();
});

settingsForm.addEventListener("change", () => {
  persistSettings();
  refreshDebounced();
});

window.addEventListener("beforeunload", () => {
  abortItemRequest();
  if (state.eventSource) {
    state.eventSource.close();
  }
});

async function initialize() {
  applySettingValues(BUILTIN_DEFAULTS);
  try {
    await loadUiConfig();
  } catch (error) {
    appendLog(`Ошибка загрузки настроек UI: ${error.message}`);
    const stored = loadStoredSettings();
    if (stored) {
      applySettingValues(stored);
    }
  }
  await loadActiveJob();
}

setTerminalState("idle");
setBusy(false);
initialize().catch((error) => {
  appendLog(`Ошибка инициализации: ${error.message}`);
});
