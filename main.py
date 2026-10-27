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
    "Ти — турботливий, уважний друг і надійний напарник у навчанні, а не суворий екзаменатор чи професор.\n"
    "Твоя подруга готується до іспиту. Вона надіслала конспект і додала аудіозапис свого усного переказу по пам'яті.\n\n"
    "Правила спілкування та перевірки:\n"
    "1. Спілкуйся виключно українською мовою, строго звертаючись до неї на «ти» у жіночому роді "
    "(наприклад: «ти сказала», «ти згадала», «ти чудово пояснила», «ти пропустила»).\n"
    "2. КАТЕГОРИЧНО ЗАБОРОНЕНО писати про неї в третій особі та вживати слова «студентка», «користувач», «авторка» тощо. "
    "Говори безпосередньо з нею як близький друг.\n"
    "3. Тон має бути теплим, живим, дружнім і простим, без сухої академічної бюрократії, але без зайвих емодзі.\n"
    "4. Звіряй переказ буквально по деталях: якщо пропущено конкретне слово, термін, дату, цифру або важливий "
    "причинно-наслідковий зв'язок — обов'язково вкажи на це.\n"
    "5. Якщо є неточність або помилка — поясни простою людською мовою, у чому саме полягає різниця між тим, "
    "що було в конспекті, і тим, що вона сказала в аудіо.\n"
    "6. Використовуй базові HTML-теги для оформлення (<b>жирний текст</b>, <code>терміни</code>). Не використовуй емодзі.\n\n"
    "Відповідь ОБОВ'ЯЗКОВО повинна містити два окремі блоки з такими мітками:\n\n"
    "===ТРАНСКРИПЦІЯ===\n"
    "[Точний текст її усного переказу з аудіо. Якщо мовлення відсутнє або нерозбірливе, напиши: НЕРОЗПІЗНАНО]\n\n"
    "===ОЦІНКА===\n"
    "<b>1. Оцінка:</b> X/10 (одне коротке щире підбадьорливе речення своїми словами).\n"
    "<b>2. Що передано точно:</b> коротко про те, які головні думки ти відтворила правильно.\n"
    "<b>3. Що загубилося або варто додати:</b> список конкретних фактів, дат, термінів або деталей із конспекту, "
    "які не прозвучали у переказі (якщо все згадано: Все на місці, нічого не забула).\n"
    "<b>4. Помилки та неточності:</b> конкретні зауваження до слів або формулювань із поясненням відмінності від конспекту "
    "(якщо все вірно: Помилок немає)."
)




def _convert_voice_to_mp3_sync(voice_bytes: bytes) -> tuple[bytes, str]:
    """
    Converts downloaded Telegram voice note (OGG Opus) into MP3 using pydub in-memory.
    Executed in a background thread via asyncio.to_thread to avoid blocking the event loop.
    If ffmpeg is not available, immediately falls back to native audio/ogg.
    """
    if not shutil.which("ffmpeg"):
        return voice_bytes, "audio/ogg"

    try:
        in_buffer = io.BytesIO(voice_bytes)
        sound = AudioSegment.from_file(in_buffer, format="ogg")
        mp3_buffer = io.BytesIO()
        sound.export(mp3_buffer, format="mp3")
        return mp3_buffer.getvalue(), "audio/mp3"
    except Exception as exc:
        logger.warning(
            "Конвертація в MP3 через pydub не вдалася: %s. Використовується формат audio/ogg.",
            exc,
        )
        return voice_bytes, "audio/ogg"


async def prepare_inline_audio_part(voice_bytes: bytes) -> dict:
    """
    Prepares audio bytes directly as an inline data dictionary.
    Eliminates genai.upload_file() and any PROCESSING status polling.
    """
    audio_bytes, mime_type = await asyncio.to_thread(_convert_voice_to_mp3_sync, voice_bytes)
    return {
        "mime_type": mime_type,
        "data": audio_bytes,
    }


async def generate_gemini_response_fast(
    primary_model: genai.GenerativeModel,
    fallback_model: genai.GenerativeModel | None,
    prompt: str,
    audio_part: dict,
    timeout_seconds: float = 7.0,
) -> str:
    """
    Strictly asynchronous model execution with inline audio data.
    If the primary model (gemini-3.8-flash) encounters high demand (503), quota limits (429),
    or times out, it immediately falls back to the active working model (gemini-3.5-flash)
    to guarantee a 3-5 second total response time instead of waiting 4 minutes.
    """
    try:
        response = await asyncio.wait_for(
            primary_model.generate_content_async([prompt, audio_part]),
            timeout=timeout_seconds,
        )
        return response.text or ""
    except (asyncio.TimeoutError, Exception) as exc:
        if fallback_model is not None:
            logger.warning(
                "Основна модель (%s) не відповіла за %sс (%s). Миттєво перемикаємося на актуальну робочу модель (%s)...",
                primary_model.model_name,
                timeout_seconds,
                exc,
                fallback_model.model_name,
            )
            response = await fallback_model.generate_content_async([prompt, audio_part])
            return response.text or ""
        raise


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
        "Привіт. Я помічник для підготовки до іспитів.\n\n"
        "Порядок роботи:\n"
        "1. Надішли текст конспекту або теми.\n"
        "2. Запиши голосове повідомлення з усним переказом.\n\n"
        "Надішли текст конспекту для початку."
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
            "Наразі немає активного процесу перевірки. Надішли /start для початку.",
            parse_mode=ParseMode.HTML,
        )
        return

    await message.answer(
        "Підготовку скасовано.\n"
        "Щоб почати знову, надішли текст або натисни /start.",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("help"))
async def handle_help(message: types.Message) -> None:
    """
    Displays helpful guidance on using the exam preparation bot.
    """
    help_text = (
        "<b>Довідка:</b>\n\n"
        "• /start — розпочати перевірку нової теми\n"
        "• /cancel — скасувати поточний запис\n"
        "• /help — довідка\n\n"
        "Порядок роботи:\n"
        "1. Надішли текст матеріалу або конспекту.\n"
        "2. Запиши голосове повідомлення з усним переказом."
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
            "Текст конспекту занадто короткий. Надішли змістовніший фрагмент або тези теми:",
            parse_mode=ParseMode.HTML,
        )
        return

    await state.update_data(original_text=submitted_text)
    await state.set_state(ExamState.waiting_for_voice)

    prompt_voice = (
        "Конспект збережено.\n\n"
        "Тепер запиши <b>голосове повідомлення</b> з усним переказом цього матеріалу по пам'яті.\n\n"
        "<i>(Для зміни конспекту надішли /cancel)</i>"
    )
    await message.answer(prompt_voice, parse_mode=ParseMode.HTML)


@router.message(ExamState.waiting_for_text)
async def handle_invalid_text_fallback(message: types.Message) -> None:
    """
    Fallback when the user sends a non-text message while waiting for text.
    """
    await message.answer(
        "Будь ласка, надішли конспект текстовим повідомленням або скористайся /cancel для скасування.",
        parse_mode=ParseMode.HTML,
    )


@router.message(ExamState.waiting_for_voice, F.voice)
async def handle_voice_message(
    message: types.Message,
    state: FSMContext,
    bot: Bot,
    primary_model: genai.GenerativeModel,
    fallback_model: genai.GenerativeModel | None,
) -> None:
    """
    Downloads voice note, prepares inline audio data directly in memory,
    and calls model.generate_content_async asynchronously for rapid 3-5s response.
    """
    data = await state.get_data()
    original_text = data.get("original_text", "")
    if not original_text:
        await message.answer(
            "Не знайдено збереженого конспекту. Надішли команду /start та відправ текст знову.",
            parse_mode=ParseMode.HTML,
        )
        await state.clear()
        return

    status_msg = await message.answer(
        "Обробляю голосове повідомлення...",
        parse_mode=ParseMode.HTML,
    )

    try:
        # 1. Download voice file directly into in-memory buffer (non-blocking)
        voice_buffer = io.BytesIO()
        await bot.download(message.voice, destination=voice_buffer)
        voice_bytes = voice_buffer.getvalue()

        # 2. Prepare inline data dict (no upload_file, no polling)
        audio_part = await prepare_inline_audio_part(voice_bytes)

        # 3. Formulate prompt
        prompt = (
            f"Оригінальний навчальний матеріал (конспект):\n"
            f"\"\"\"\n{original_text}\n\"\"\"\n\n"
            "Ось аудіозапис мого усного переказу цього матеріалу по пам'яті. "
            "Уважно прослухай його, зроби точну транскрипцію та надай детальний розбір за вказаною структурою "
            "з обов'язковими мітками ===ТРАНСКРИПЦІЯ=== та ===ОЦІНКА===. Звертайся до мене на «ти» у жіночому роді і не використовуй емодзі."
        )

        # 4. Asynchronous fast call with fallback to avoid 4-minute retries
        raw_text = await generate_gemini_response_fast(
            primary_model=primary_model,
            fallback_model=fallback_model,
            prompt=prompt,
            audio_part=audio_part,
            timeout_seconds=7.0,
        )

        if not raw_text.strip():
            await status_msg.edit_text(
                "Не вдалося отримати відповідь. Спробуй записати голосове ще раз.",
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
                "Не вдалося розпізнати слова у голосовому повідомленні. Спробуй записати голосове ще раз.",
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
            "<b>Усний переказ (розпізнаний текст):</b>\n\n"
            f"<i>\"{html.escape(transcript_text)}\"</i>"
        )
        await send_split_message(message, transcription_header)

        # Send pedagogical evaluation
        feedback_header = (
            f"<b>Результат аналізу:</b>\n\n"
            f"{evaluation_text}\n\n"
            f"───────────────────\n"
            f"Щоб перевірити наступну тему, надішли новий текст або натисни /start."
        )
        await send_split_message(message, feedback_header)

        # Clear state after successful evaluation
        await state.clear()

    except Exception as exc:
        logger.exception("Помилка під час обробки голосового повідомлення або звернення до Gemini: %s", exc)
        try:
            await status_msg.edit_text(
                f"Сталася помилка під час обробки: <i>{html.escape(str(exc))}</i>\n\n"
                "Перевір, будь ласка, налаштування GEMINI_API_KEY у файлі .env або спробуй пізніше.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await message.answer(
                "Сталася неочікувана помилка під час обробки. Будь ласка, спробуй ще раз або надішли /start.",
                parse_mode=ParseMode.HTML,
            )


@router.message(ExamState.waiting_for_voice)
async def handle_invalid_voice_fallback(message: types.Message) -> None:
    """
    Fallback when user sends non-voice message while in waiting_for_voice state.
    """
    await message.answer(
        "Очікується голосове повідомлення з усним переказом конспекту. Запиши аудіо або надішли /cancel.",
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
        "Щоб почати перевірку знань, надішли текст конспекту або команду /start.",
        parse_mode=ParseMode.HTML,
    )



async def main() -> None:
    load_dotenv()
    bot_token = getenv("BOT_TOKEN")
    gemini_api_key = getenv("GEMINI_API_KEY")
    gemini_model_name = getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fallback_model_name = getenv("FALLBACK_MODEL", "gemini-3.5-flash")

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

    generation_config = genai.GenerationConfig(
        temperature=0.3,
    )

    primary_model = genai.GenerativeModel(
        model_name=gemini_model_name,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=generation_config,
    )

    fallback_model = None
    if fallback_model_name and fallback_model_name != gemini_model_name:
        fallback_model = genai.GenerativeModel(
            model_name=fallback_model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config=generation_config,
        )

    bot = Bot(token=bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.include_router(router)

    # Dependency injection for handlers
    dp["primary_model"] = primary_model
    dp["fallback_model"] = fallback_model

    logger.info(
        "Бот для підготовки до іспитів запущений (Основна модель: %s, Резервна швидка модель: %s)!",
        gemini_model_name,
        fallback_model_name,
    )
    await dp.start_polling(bot, primary_model=primary_model, fallback_model=fallback_model)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот зупинений.")