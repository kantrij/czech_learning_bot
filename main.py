import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

import asyncio
import html
import io
import logging
from os import getenv
import shutil
import sys

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import google.generativeai as genai
from pydub import AudioSegment


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

# FSM states for the exam preparation workflow
class ExamState(StatesGroup):
    waiting_for_text = State()
    waiting_for_voice = State()

router = Router()

SYSTEM_INSTRUCTION = (
    "Ти — професійний, доброзичливий, уважний і турботливий викладач-ментор. "
    "Студентка готується до важливого іспиту. Вона надіслала конспект (навчальний матеріал) "
    "і додала аудіозапис свого усного переказу з пам'яті.\n\n"
    "Критичні педагогічні настанови:\n"
    "1. Уважно прослухай аудіозапис та зроби його точну розшифровку (транскрипцію) українською мовою.\n"
    "2. Оцінюй семантичний зміст і розуміння концепцій, а не дослівне цитування чи точний збіг слів. "
    "Студентка висловлює думки своїми словами.\n"
    "3. Відповідай виключно українською мовою.\n"
    "4. Зберігай теплий, підбадьорливий, щирий і водночас об'єктивно конструктивний тон.\n"
    "5. Використовуй HTML-теги для форматування (<b>жирний текст</b>, <i>курсив</i>, <code>терміни</code>).\n\n"
    "Твоя відповідь ОБОВ'ЯЗКОВО повинна містити два окремі блоки з такими мітками:\n\n"
    "===ТРАНСКРИПЦІЯ===\n"
    "[Тут запиши точну транскрипцію усного переказу студентки. Якщо мовлення відсутнє або неможливо розібрати слова, напиши: НЕРОЗПІЗНАНО]\n\n"
    "===ОЦІНКА===\n"
    "<b>1. 🎯 Загальна оцінка:</b> від 1 до 10 балів із коротким теплим коментарем.\n"
    "<b>2. 💡 Засвоєння ключових концепцій:</b> які головні ідеї розкриті правильно та влучно.\n"
    "<b>3. 🔍 Що варто додати:</b> важливі поняття, терміни, факти або нюанси з конспекту, які були пропущені.\n"
    "<b>4. ⚠️ Неточності чи помилки:</b> фактичні викривлення чи неточності (якщо помилок немає — відзнач це і похвали за точність!).\n"
    "<b>5. 🌟 Підбадьорливий висновок:</b> теплі слова підтримки та мотивація перед іспитом."
)


def _convert_ogg_to_mp3_sync(voice_bytes: bytes) -> tuple[bytes, str]:
    """
    Synchronous conversion of OGG Opus to MP3 using pydub.
    Executed inside asyncio.to_thread to prevent blocking the event loop.
    If ffmpeg is not available, immediately falls back to native audio/ogg.
    """
    if not shutil.which("ffmpeg"):
        return voice_bytes, "audio/ogg"

    try:
        in_buffer = io.BytesIO(voice_bytes)
        sound = AudioSegment.from_file(in_buffer, format="ogg")
        out_buffer = io.BytesIO()
        sound.export(out_buffer, format="mp3")
        return out_buffer.getvalue(), "audio/mp3"
    except Exception as exc:
        logger.warning(
            "Конвертація в MP3 через pydub не вдалася: %s. Використовується формат audio/ogg.",
            exc,
        )
        return voice_bytes, "audio/ogg"


async def prepare_audio_inline_data(voice_bytes: bytes) -> dict:
    """
    Prepares audio bytes non-blockingly and returns inline data dict for Gemini.
    Avoids using Google File API upload to ensure zero upload delay.
    """
    audio_bytes, mime_type = await asyncio.to_thread(_convert_ogg_to_mp3_sync, voice_bytes)
    return {
        "mime_type": mime_type,
        "data": audio_bytes,
    }


async def send_split_message(
    target_message: types.Message,
    text: str,
    parse_mode: str | None = ParseMode.HTML,
) -> None:
    """
    Safely delivers message content to Telegram, splitting long text (>4000 characters)
    by line breaks and falling back to unformatted text if HTML parsing encounters issues.
    """
    max_chunk_size = 4000
    if len(text) <= max_chunk_size:
        try:
            await target_message.answer(text, parse_mode=parse_mode)
        except TelegramBadRequest:
            await target_message.answer(text, parse_mode=None)
        return

    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            while len(line) > max_chunk_size:
                chunks.append(line[:max_chunk_size])
                line = line[max_chunk_size:]
        current_chunk.append(line)
        current_len += len(line) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    for chunk in chunks:
        if chunk.strip():
            try:
                await target_message.answer(chunk, parse_mode=parse_mode)
            except TelegramBadRequest:
                await target_message.answer(chunk, parse_mode=None)


@router.message(CommandStart())
async def handle_start(message: types.Message, state: FSMContext) -> None:
    """
    Greets the user in Ukrainian, resets any existing FSM state,
    and invites them to send study materials.
    """
    await state.clear()
    await state.set_state(ExamState.waiting_for_text)

    welcome_text = (
        "Привіт! 🌸 Я твій помічник для підготовки до іспитів на базі <b>Google Gemini AI</b>.\n\n"
        "<b>Як ми готуємося:</b>\n"
        "1️⃣ Надішли мені <b>текст конспекту</b>, лекції або тему, яку ти вчиш 📖\n"
        "2️⃣ Запиши <b>голосове повідомлення 🎙</b>, де своїми словами по пам'яті розкажи цей матеріал.\n"
        "3️⃣ Я швидко прослухаю аудіо, транскрибую його, порівняю з конспектом і надам детальну оцінку з теплими порадами! ✨\n\n"
        "Чекаю на текст конспекту! 📚"
    )
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


@router.message(Command("cancel", "reset"))
async def handle_cancel(message: types.Message, state: FSMContext) -> None:
    """
    Allows the user to cancel the current session and reset state.
    """
    current_state = await state.get_state()
    await state.clear()
    if current_state is None:
        await message.answer(
            "Наразі немає активного процесу перевірки. Надішли /start, щоб розпочати! 💫",
            parse_mode=ParseMode.HTML,
        )
        return

    await message.answer(
        "Поточну підготовку скасовано 🔄.\n"
        "Коли будеш готова продовжити — просто надішли новий текст конспекту або натисни /start!",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("help"))
async def handle_help(message: types.Message) -> None:
    """
    Displays helpful guidance on using the exam preparation bot.
    """
    help_text = (
        "🎓 <b>Довідка помічника для іспитів:</b>\n\n"
        "• <b>/start</b> — розпочати нове тренування теми\n"
        "• <b>/cancel</b> — скасувати поточний запис та скинути стан\n"
        "• <b>/help</b> — переглянути цю довідку\n\n"
        "<b>Як це працює:</b>\n"
        "1. Відправ текст матеріалу або конспекту.\n"
        "2. Запиши голосове повідомлення з усним переказом з пам'яті.\n"
        "3. Gemini AI оцінить відповідь від 1 до 10, виділить ключові ідеї, пропущені факти та надасть мотивуючі рекомендації!"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)


@router.message(ExamState.waiting_for_text, F.text)
async def handle_study_text(message: types.Message, state: FSMContext) -> None:
    """
    Receives and stores study material in FSM storage,
    then transitions state to waiting_for_voice.
    """
    submitted_text = (message.text or "").strip()
    if len(submitted_text) < 10:
        await message.answer(
            "Текст конспекту здається занадто коротким 🤏. "
            "Будь ласка, надішли більш детальний матеріал або тези теми:",
            parse_mode=ParseMode.HTML,
        )
        return

    await state.update_data(original_text=submitted_text)
    await state.set_state(ExamState.waiting_for_voice)

    prompt_voice = (
        "Чудово! Конспект надійно збережено 📖\n\n"
        "Тепер запиши <b>голосове повідомлення 🎙</b>, де своїми словами розкажи цей матеріал по пам'яті.\n"
        "Говори спокійно й упевнено — це чудове тренування перед іспитом! ✨\n\n"
        "<i>(Якщо захочеш замінити текст конспекту — надішли /cancel)</i>"
    )
    await message.answer(prompt_voice, parse_mode=ParseMode.HTML)


@router.message(ExamState.waiting_for_text)
async def handle_invalid_text_fallback(message: types.Message) -> None:
    """
    Fallback when the user sends a non-text message while waiting for text.
    """
    await message.answer(
        "Будь ласка, надішли конспект саме <b>текстовим повідомленням</b> 📝 "
        "або скористайся командою /cancel для скасування.",
        parse_mode=ParseMode.HTML,
    )


@router.message(ExamState.waiting_for_voice, F.voice)
async def handle_voice_message(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    gemini_model_instance: genai.GenerativeModel,
) -> None:
    """
    Downloads voice note, prepares inline audio data asynchronously without blocking the event loop,
    and calls model.generate_content_async directly.
    """
    data = await state.get_data()
    original_text = data.get("original_text", "")
    if not original_text:
        await message.answer(
            "⚠️ Не вдалося знайти збережений конспект. "
            "Будь ласка, надішли команду /start та відправ текст знову.",
            parse_mode=ParseMode.HTML,
        )
        await state.clear()
        return

    # Intermediate status update: Audio processing & Gemini evaluation
    status_msg = await message.answer(
        "🎧 Слухаю голосове та аналізую відповідь через Gemini AI...",
        parse_mode=ParseMode.HTML,
    )

    try:
        # 1. Download voice file into memory buffer (non-blocking)
        voice_buffer = io.BytesIO()
        await bot.download(message.voice, destination=voice_buffer)
        voice_bytes = voice_buffer.getvalue()

        # 2. Prepare audio inline data in background thread without blocking event loop
        audio_part = await prepare_audio_inline_data(voice_bytes)

        # 3. Formulate prompt
        prompt = (
            f"Оригінальний навчальний матеріал (конспект):\n"
            f"\"\"\"\n{original_text}\n\"\"\"\n\n"
            "Студентка записала аудіо зі своїм переказом цього конспекту по пам'яті. "
            "Будь ласка, прослухай аудіо, транскрибуй його та надай детальний педагогічний аналіз за вказаною структурою "
            "з обов'язковими мітками ===ТРАНСКРИПЦІЯ=== та ===ОЦІНКА===."
        )

        # 4. Strictly asynchronous Gemini call using inline audio data
        response = await gemini_model_instance.generate_content_async([prompt, audio_part])

        try:
            raw_text = response.text or ""
        except (ValueError, AttributeError):
            raw_text = ""

        if not raw_text.strip():
            await status_msg.edit_text(
                "⚠️ Не вдалося отримати відповідь від Gemini. Спробуй записати голосове ще раз 🎙",
                parse_mode=ParseMode.HTML,
            )
            return

        # 5. Parse delimiters
        if "===ТРАНСКРИПЦІЯ===" in raw_text and "===ОЦІНКА===" in raw_text:
            after_trans = raw_text.split("===ТРАНСКРИПЦІЯ===")[1]
            parts = after_trans.split("===ОЦІНКА===")
            transcript_text = parts[0].strip()
            evaluation_text = parts[1].strip() if len(parts) > 1 else ""
        else:
            transcript_text = "Розпізнано з аудіо"
            evaluation_text = raw_text.strip()

        if not transcript_text or "НЕРОЗПІЗНАНО" in transcript_text.upper():
            await status_msg.edit_text(
                "⚠️ Не вдалося розпізнати слова у голосовому повідомленні. "
                "Можливо, запис був надто тихим або перервався. Спробуй записати голосове ще раз 🎙",
                parse_mode=ParseMode.HTML,
            )
            return

        # Cleanup status message
        try:
            await status_msg.delete()
        except Exception:
            pass

        # Send transcribed speech
        transcription_header = (
            "📝 <b>Твій усний переказ (розпізнаний текст):</b>\n\n"
            f"<i>\"{html.escape(transcript_text)}\"</i>"
        )
        await send_split_message(message, transcription_header)

        # Send pedagogical evaluation
        feedback_header = (
            f"📊 <b>Результат педагогічного аналізу:</b>\n\n"
            f"{evaluation_text}\n\n"
            f"───────────────────\n"
            f"🌸 <i>Ти молодець! Щоб потренувати наступну тему, просто надішли новий текст або натисни /start.</i>"
        )
        await send_split_message(message, feedback_header)

        # Clear state after successful evaluation
        await state.clear()

    except Exception as exc:
        logger.exception("Помилка під час обробки голосового повідомлення або звернення до Gemini: %s", exc)
        try:
            await status_msg.edit_text(
                f"❌ Сталася помилка під час обробки: <i>{html.escape(str(exc))}</i>\n\n"
                "Перевір, будь ласка, чи дійсний ключ <code>GEMINI_API_KEY</code> у файлі <code>.env</code> або спробуй пізніше.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await message.answer(
                "❌ Сталася неочікувана помилка під час обробки. Будь ласка, спробуй ще раз або надішли /start.",
                parse_mode=ParseMode.HTML,
            )


@router.message(ExamState.waiting_for_voice)
async def handle_invalid_voice_fallback(message: types.Message) -> None:
    """
    Fallback when user sends non-voice message while in waiting_for_voice state.
    """
    await message.answer(
        "Зараз я очікую саме <b>голосове повідомлення 🎙</b> з твоїм усним переказом конспекту.\n"
        "Будь ласка, запиши аудіо або надішли /cancel, якщо хочеш розпочати спочатку.",
        parse_mode=ParseMode.HTML,
    )


@router.message(F.text)
async def handle_text_outside_state(message: types.Message, state: FSMContext) -> None:
    """
    Conveniently accepts study text even if the user didn't explicitly execute /start first.
    """
    current_state = await state.get_state()
    if current_state is None:
        await state.set_state(ExamState.waiting_for_text)
        await handle_study_text(message, state)


@router.message()
async def handle_generic_fallback(message: types.Message) -> None:
    """
    Default fallback for unrecognized inputs when no state is active.
    """
    await message.answer(
        "Привіт! 🌸 Щоб почати підготовку до іспиту, надішли мені <b>текст конспекту</b> або команду /start 📚",
        parse_mode=ParseMode.HTML,
    )


async def main() -> None:
    load_dotenv()
    bot_token = getenv("BOT_TOKEN")
    gemini_api_key = getenv("GEMINI_API_KEY")
    gemini_model = getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if not bot_token:
        logger.critical("BOT_TOKEN не знайдено! Перевірте наявність BOT_TOKEN у файлі .env.")
        sys.exit(1)

    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        logger.warning(
            "GEMINI_API_KEY не задано або містить шаблонне значення! "
            "Переконайтеся, що ви додали дійсний ключ GEMINI_API_KEY у файл .env для роботи Gemini AI."
        )

    # Configure Google Generative AI
    genai.configure(api_key=gemini_api_key or "missing-key")
    model = genai.GenerativeModel(
        model_name=gemini_model,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
        ),
    )

    bot = Bot(token=bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.include_router(router)

    # Dependency injection for handlers
    dp["gemini_model_instance"] = model

    logger.info(
        "Бот для підготовки до іспитів на базі Google Gemini (%s) успішно запущений!",
        gemini_model,
    )
    await dp.start_polling(bot, gemini_model_instance=model)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот зупинений.")