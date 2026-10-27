# Exam Preparation Assistant Bot (Telegram + Google Gemini AI)

Telegram-бот на базі **aiogram 3.x** та **Google Gemini API**, розроблений для допомоги у підготовці до іспитів.

### Як це працює:
1. Користувачка надсилає текст конспекту або теми (`/start` -> надсилання тексту).
2. Записує голосове повідомлення, переказуючи матеріал своїми словами по пам'яті.
3. Бот передає аудіо як inline data разом із текстом конспекту в модель Gemini.
4. Gemini розпізнає переказ і формує чіткий діловий аналіз за 4 пунктами:
   - Загальна оцінка (X/10).
   - Засвоєння ключових концепцій.
   - Що варто додати (пропущені деталі/терміни).
   - Неточності чи помилки (якщо є).

---

## Системні залежності

### FFmpeg (Опціонально)
Для попередньої конвертації аудіо в MP3 використовується `pydub`.
*(Примітка: якщо `ffmpeg` не встановлено, бот автоматично надсилає нативний буфер OGG Opus (`audio/ogg`) прямо в Gemini API, оскільки Gemini підтримує аудіо OGG Opus нативно).*

- **Windows**: `winget install Gyan.FFmpeg` або `choco install ffmpeg`
- **Ubuntu / Debian**: `sudo apt update && sudo apt install -y ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## Встановлення

1. **Клонуйте репозиторій або перейдіть у робочу директорію:**
   ```bash
   cd czech_learning_bot
   ```

2. **Створіть та активуйте віртуальне оточення:**
   - Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Встановіть залежності:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Налаштування (.env)

Створіть або відредагуйте файл `.env` у корені проєкту:

```env
BOT_TOKEN=ваш_telegram_bot_token_від_BotFather
GEMINI_API_KEY=ваш_ключ_з_Google_AI_Studio
GEMINI_MODEL=gemini-3.8-flash
FALLBACK_MODEL=gemini-3.5-flash
```

---

## Запуск бота

```bash
python main.py
```

---

## Команди бота

- `/start` — Розпочати перевірку нової теми.
- `/cancel` — Скасувати поточний запис та скинути стан.
- `/help` — Показати довідку.