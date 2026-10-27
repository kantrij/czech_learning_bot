# Exam Preparation Assistant Bot (Telegram + AI) 🎓

Telegram-бот на базі **aiogram 3.x**, розроблений для допомоги у підготовці до іспитів. 

### 💡 Як це працює:
1. Користувачка надсилає текст конспекту або теми (`/start` -> надсилання тексту).
2. Записує голосове повідомлення, переказуючи матеріал своїми словами по пам'яті.
3. Бот завантажує голосове повідомлення, конвертує його та транскрибує через **OpenAI Whisper API** (з підсиленням термінології з тексту конспекту).
4. Порівнює усний переказ із конспектом за допомогою **GPT-4o-mini**, оцінюючи семантичне розуміння, а не дослівний збіг.
5. Надає структурований, теплий та мотивуючий відгук українською мовою з оцінкою від 1 до 10, виділенням ключових ідей, пропущених фактів та корисними порадами.

---

## 🛠 Системні залежності (System Dependencies)

### FFmpeg
Для перекодування аудіо з формату Telegram Voice (`.ogg` Opus) у `.mp3` використовується бібліотека `pydub`, яка спирається на системну утиліту `ffmpeg`:
*(Примітка: якщо `ffmpeg` тимчасово не встановлено, бот автоматично надішле нативний OGG Opus у Whisper API, оскільки API OpenAI також підтримує `.ogg`).*

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
   *(Для Python 3.13+ автоматично встановлюється `audioop-lts` для сумісності з `pydub`).*

---

## ⚙️ Налаштування (.env)

Створіть або відредагуйте файл `.env` у корені проєкту:

```env
BOT_TOKEN=ваш_telegram_bot_token_від_BotFather
OPENAI_API_KEY=ваш_openai_api_key
```

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