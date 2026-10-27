import asyncio
import html
import io
import logging
from os import getenv
import sys

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from openai import AsyncOpenAI
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


def prepare_audio_for_stt(voice_bytes: bytes) -> tuple[str, bytes, str]:
    """
    Attempts to convert raw voice bytes (Ogg Opus from Telegram) to MP3 using pydub.
    If ffmpeg is missing or conversion fails, safely falls back to native OGG Opus,
    which is natively supported by the OpenAI Whisper API.

    Returns:
        tuple of (filename, audio_bytes, content_type)
    """
    try:
        in_buffer = io.BytesIO(voice_bytes)
        sound = AudioSegment.from_file(in_buffer, format="ogg")
        out_buffer = io.BytesIO()
        sound.export(out_buffer, format="mp3")
        return "voice.mp3", out_buffer.getvalue(), "audio/mpeg"
    except Exception as exc:
        logger.warning(
            "Конвертація через pydub не вдалася (можливо, ffmpeg не встановлено): %s. "
            "Використовується оригінальний формат OGG Opus для Whisper API.",
            exc,
        )
        return "voice.ogg", voice_bytes, "audio/ogg"


async def transcribe_voice(
    openai_client: AsyncOpenAI,
    voice_bytes: bytes,
    reference_text: str | None = None,
) -> str:
    """
    Transcribes audio using OpenAI Whisper API (whisper-1) in Ukrainian.
    Biases domain vocabulary by passing the initial snippet of the reference text in prompt.
    """
    filename, audio_bytes, content_type = prepare_audio_for_stt(voice_bytes)

    # Bias terminology recognition using up to ~224 tokens (~350 characters)
    prompt_bias: str | None = None
    if reference_text:
        prompt_bias = reference_text[:350].strip()

    transcription = await openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes, content_type),
        language="uk",
        prompt=prompt_bias,
    )
    return transcription.text.strip()


async def evaluate_material(
    openai_client: AsyncOpenAI,
    original_text: str,
    voice_transcript: str,
) -> str:
    """
    Compares the student's verbal explanation against the source study notes
    using gpt-4o-mini and generates structured, encouraging pedagogical feedback in Ukrainian.
    """
    system_instruction = (
        "Ти — професійний, доброзичливий, турботливий викладач-ментор. "
        "Твоє завдання — оцінити усний переказ навчального матеріалу студентки з пам'яті, "
        "порівнявши його з оригінальним конспектом.\n\n"
        "Критичні педагогічні настанови:\n"
        "1. Оцінюй семантичний зміст і розуміння концепцій, а не дослівне цитування чи точний збіг слів. "
        "Студентка висловлює думки своїми словами.\n"
        "2. Відповідай виключно українською мовою.\n"
        "3. Зберігай теплий, підбадьорливий, щирий і водночас об'єктивно конструктивний тон.\n"
        "4. Використовуй HTML-теги для форматування (<b>жирний текст</b>, <i>курсив</i>, <code>терміни</code>).\n\n"
        "Обов'язково чітко структуруй свою відповідь за такими 5 розділами:\n"
        "<b>1. 🎯 Загальна оцінка:</b> від 1 до 10 балів із коротким теплим коментарем.\n"
        "<b>2. 💡 Засвоєння ключових концепцій:</b> які головні ідеї розкриті правильно та влучно.\n"
        "<b>3. 🔍 Що варто додати:</b> важливі поняття, терміни, факти або деталі з конспекту, які були пропущені.\n"
        "<b>4. ⚠️ Неточності чи помилки:</b> фактичні викривлення чи неточності (якщо помилок немає — відзнач це і щиро похвали за точність!).\n"
        "<b>5. 🌟 Підбадьорливий висновок:</b> теплі слова підтримки та мотивація перед іспитом."
    )

    user_prompt = (
        f"<b>Оригінальний навчальний матеріал (конспект):</b>\n"
        f"<blockquote>{html.escape(original_text)}</blockquote>\n\n"
        f"<b>Транскрипція усного переказу студентки:</b>\n"
        f"<blockquote>{html.escape(voice_transcript)}</blockquote>\n\n"
        "Будь ласка, уважно проаналізуй переказ і надай структурований зворотний зв'язок відповідно до інструкцій."
    )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


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
        "Привіт! 🌸 Я твій помічник для підготовки до іспитів.\n\n"
        "<b>Як ми готуємося:</b>\n"
        "1️⃣ Надішли мені <b>текст конспекту</b>, лекції або тему, яку ти вчиш 📖\n"
        "2️⃣ Запиши <b>голосове повідомлення 🎙</b>, де своїми словами по пам'яті розкажи цей матеріал.\n"
        "3️⃣ Я уважно послухаю, перевірю твої знання за допомогою штучного інтелекту та надам детальну оцінку з теплими порадами! ✨\n\n"
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
        "3. Отримай оцінку від 1 до 10, розбір ключових ідей, пропущених деталей та мотивуючі рекомендації!"
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
    openai_client: AsyncOpenAI,
) -> None:
    """
    Downloads voice note, transcribes with Whisper (using prompt biasing),
    evaluates semantic retention against original text with gpt-4o-mini,
    and returns comprehensive feedback in Ukrainian.
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

    # Intermediate status update 1: STT
    status_msg = await message.answer(
        "🎧 Слухаю голосове та розпізнаю мову...",
        parse_mode=ParseMode.HTML,
    )

    try:
        # Download voice file into memory buffer
        voice_buffer = io.BytesIO()
        await bot.download(message.voice, destination=voice_buffer)
        voice_bytes = voice_buffer.getvalue()

        # Step 1: Transcribe via Whisper
        transcript = await transcribe_voice(
            openai_client=openai_client,
            voice_bytes=voice_bytes,
            reference_text=original_text,
        )

        if not transcript or not transcript.strip():
            await status_msg.edit_text(
                "⚠️ Не вдалося розпізнати слова у голосовому повідомленні. "
                "Можливо, запис був надто тихим. Спробуй записати голосове ще раз 🎙",
                parse_mode=ParseMode.HTML,
            )
            return

        # Intermediate status update 2: Evaluation
        await status_msg.edit_text(
            "🧠 Аналізую переказ та порівнюю з конспектом...",
            parse_mode=ParseMode.HTML,
        )

        # Step 2: Evaluate via LLM (gpt-4o-mini)
        evaluation = await evaluate_material(
            openai_client=openai_client,
            original_text=original_text,
            voice_transcript=transcript,
        )

        # Cleanup status message
        try:
            await status_msg.delete()
        except Exception:
            pass

        # Send transcribed speech
        transcription_header = (
            "📝 <b>Твій усний переказ (розпізнаний текст):</b>\n\n"
            f"<i>\"{html.escape(transcript)}\"</i>"
        )
        await send_split_message(message, transcription_header)

        # Send pedagogical evaluation
        feedback_header = (
            f"📊 <b>Результат педагогічного аналізу:</b>\n\n"
            f"{evaluation}\n\n"
            f"───────────────────\n"
            f"🌸 <i>Ти молодець! Щоб потренувати наступну тему, просто надішли новий текст або натисни /start.</i>"
        )
        await send_split_message(message, feedback_header)

        # Clear state after successful evaluation
        await state.clear()

    except Exception as exc:
        logger.exception("Помилка під час обробки голосового повідомлення або звернення до OpenAI: %s", exc)
        try:
            await status_msg.edit_text(
                f"❌ Сталася помилка під час обробки: <i>{html.escape(str(exc))}</i>\n\n"
                "Перевір, будь ласка, чи дійсний ключ <code>OPENAI_API_KEY</code> у файлі <code>.env</code> або спробуй пізніше.",
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
    openai_api_key = getenv("OPENAI_API_KEY")

    if not bot_token:
        logger.critical("BOT_TOKEN не знайдено! Перевірте наявність BOT_TOKEN у файлі .env.")
        sys.exit(1)

    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        logger.warning(
            "OPENAI_API_KEY не задано або містить шаблонне значення! "
            "Переконайтеся, що ви додали дійсний ключ OPENAI_API_KEY у файл .env для роботи Whisper та GPT-4o-mini."
        )

    bot = Bot(token=bot_token)
    dp = Dispatcher(storage=MemoryStorage())
    openai_client = AsyncOpenAI(api_key=openai_api_key or "missing-key")

    dp.include_router(router)

    # Make openai_client accessible to handlers via dependency injection
    dp["openai_client"] = openai_client

    logger.info("Бот для підготовки до іспитів успішно запущений і готовий до роботи!")
    await dp.start_polling(bot, openai_client=openai_client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот зупинений.")