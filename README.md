# Exam Preparation Assistant Bot (Telegram + Google Gemini AI) 🎓

Telegram-бот на базі **aiogram 3.x** та **Google Gemini API** (моделі `gemini-2.0-flash` або `gemini-1.5-flash`), розроблений для допомоги у підготовці до іспитів.

### 💡 Як це працює:
1. Користувачка надсилає текст конспекту або теми (`/start` -> надсилання тексту).
2. Записує голосове повідомлення, переказуючи матеріал своїми словами по пам'яті.
3. Бот завантажує голосове повідомлення та передає його разом із текстом конспекту в мультимодальну модель **Google Gemini Flash**.
4. Gemini розпізнає усний переказ, зіставляє його з навчальним матеріалом (оцінюючи семантичне розуміння суті) і формує структурований відгук.
5. Бот надсилає окремо розпізнаний текст та детальний, теплий педагогічний розбір із балом від 1 до 10, виділенням ключових ідей і підбадьорливим словом.

---

## 🛠 Системні залежності (System Dependencies)

### FFmpeg (Опціонально)
Для попередньої обробки аудіо використовується `pydub`. 
*(Примітка: якщо `ffmpeg` не встановлено, бот автоматично надсилає нативний буфер OGG Opus (`audio/ogg`) прямо в Google Gemini API, оскільки Gemini підтримує аудіо OGG Opus нативно).*

- **Windows**:
  - Через winget: `winget install Gyan.FFmpeg`
  - Через Chocolatey: `choco install ffmpeg`
  - Або завантажте збірку з офіційного сайту [ffmpeg.org](https://ffmpeg.org/download.html) та додайте папку `bin` до змінної `PATH`.
- **Ubuntu / Debian**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```

---

## 📦 Встановлення (Installation)

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

## ⚙️ Налаштування (.env)

Створіть або відредагуйте файл `.env` у корені проєкту:

```env
BOT_TOKEN=ваш_telegram_bot_token_від_BotFather
GEMINI_API_KEY=ваш_безкоштовний_ключ_з_Google_AI_Studio
GEMINI_MODEL=gemini-2.0-flash
```

> **Де взяти безкоштовний GEMINI_API_KEY:**
> Отримайте безкоштовний ключ на [Google AI Studio](https://aistudio.google.com/apikey).

---

## 🚀 Запуск бота

```bash
python main.py
```

---

## 📱 Команди бота

- `/start` — Розпочати нову сесію тренування (скидає попередній стан і запитує конспект).
- `/cancel` — Скасувати поточну сесію та скинути активний стан.
- `/help` — Показати довідку та інструкцію з використання.