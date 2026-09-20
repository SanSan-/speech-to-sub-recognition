# Инженерные стандарты

Обязательны [общий профиль](../specs/engineering-governance/reference/python-common.md)
и [проектное приложение](../specs/engineering-governance/reference/project-specific.md).

Ключевые правила: один сервис для интерфейсов; явная UTF-8 без BOM; точные зависимости в существующих requirements;
малые функции с одной ответственностью; синхронизация принятого контракта вместе с кодом. Изменение пакетного менеджера
требует отдельного обоснования.

Команды из корня проекта:

```powershell
openspec validate --all --strict --no-interactive
.\.venv\Scripts\python.exe -B -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pip check
node --check speech_to_sub\web\static\app.js
npm test
```

Установка, тяжёлые проверки, Sonar и сборка имеют отдельные условия в проектном приложении. Смысл проверок и критерии
результата — в [06](06-testing-and-quality.md).
