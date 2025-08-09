"""
Telegram bot that
  • echoes text or voice notes and returns the full transcription, and
  • lets the user *manually* choose one or more tags via inline buttons.

Once the user presses **Done**, the message together with the chosen tags is
saved to the database.

All automatic embedding‑based classification has been removed.
"""

import os
import logging
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Set

from dotenv import load_dotenv
from openai import OpenAI
import psycopg2

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
	)

# ------------------------------------------------------------------
# 🔐  Tokens & keys
# ------------------------------------------------------------------
load_dotenv()

TOKEN         = os.getenv("BOT_TOKEN")
OPEN_AI_TOKEN = os.getenv("OPENAI_API_KEY")
DB_URL        = os.getenv("SUPABASE_DB_URL")

if not all([TOKEN, OPEN_AI_TOKEN, DB_URL]):
    raise RuntimeError("Missing BOT_TOKEN, OPENAI_API_KEY or SUPABASE_DB_URL – check your .env file!")

client = OpenAI(api_key=OPEN_AI_TOKEN)

# ------------------------------------------------------------------
# Optional (but useful) logging
# ------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# ------------------------------------------------------------------
conn = psycopg2.connect(DB_URL)    # sslmode comes from the URI
conn.autocommit = True

def store_message(telegram_id: int,
                  text: str,
                  ts: datetime,
                  tags: List[str]) -> None:
    """Insert one diary entry; ignore duplicates."""
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into diary_messages (telegram_id, ts, text, tags)
            values (%s,%s,%s,%s)
            """,
            (telegram_id, ts, text, tags)
        )

# ------------------------------------------------------------------
# Tags (key → description)
# ------------------------------------------------------------------
TAGS = {
    "Tag1": "Desc 1",
    "Tag2": "Desc 2",
    "Tag3": "Desc 3",
    "Tag4": "Desc 4",
}
# ------------------------------------------------------------------
# Inline‑keyboard helpers
# ------------------------------------------------------------------

TAG_KEYS = list(TAGS.keys())

def build_keyboard(selected: Set[str]) -> InlineKeyboardMarkup:
    """Return an InlineKeyboard with tag toggles + Done button."""
    buttons: List[List[InlineKeyboardButton]] = []
    # Two tag buttons per row
    for i in range(0, len(TAG_KEYS), 2):
        row: List[InlineKeyboardButton] = []
        for key in TAG_KEYS[i:i + 2]:
            prefix = "✅ " if key in selected else ""
            row.append(
                InlineKeyboardButton(
                    text=f"{prefix}{key.replace('_', ' ')}",
                    callback_data=f"tag:{key}"
                )
            )
        buttons.append(row)
    # Final row with Done
    buttons.append([InlineKeyboardButton(text="💾 Done", callback_data="done")])
    return InlineKeyboardMarkup(buttons)

# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the incoming text and ask the user to pick tags."""
    user = update.effective_user
    text = update.message.text
    ts   = datetime.now(timezone.utc)

    context.user_data["pending_message"] = {"text": text, "ts": ts}
    context.user_data["selected_tags"]   = set()

    reply = (
        f"This is the text that {user.username or user.id} sent:\n\n"
        f"{text}\n\n"
        "Select the tags that apply and press *Done*."
    )
    await update.message.reply_text(
        reply,
        reply_markup=build_keyboard(set()),
        parse_mode="Markdown"
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download audio, transcribe with Whisper, and ask for tags."""
    user    = update.effective_user
    tg_file = await (update.message.voice or update.message.audio).get_file()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
        tmp_path = Path(tmp.name)
        await tg_file.download_to_drive(custom_path=str(tmp_path))

    def transcribe(path: Path) -> str:
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return resp.text.strip()

    try:
        transcript = await asyncio.to_thread(transcribe, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)   # clean up temp file

    ts = datetime.now(timezone.utc)
    context.user_data["pending_message"] = {"text": transcript, "ts": ts}
    context.user_data["selected_tags"]   = set()

    reply = (
        f"This is the transcription that {user.username or user.id} sent:\n\n"
        f"{transcript}\n\n"
        "Select the tags that apply and press *Done*."
    )
    await update.message.reply_text(
        reply,
        reply_markup=build_keyboard(set()),
        parse_mode="Markdown"
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle tag selection and final save."""
    query  = update.callback_query
    await query.answer()
    user   = update.effective_user

    if "pending_message" not in context.user_data:
        await query.edit_message_text("Nothing to store – please send a new message first.")
        return

    selected: Set[str] = context.user_data.get("selected_tags", set())
    data = query.data

    if data.startswith("tag:"):
        tag = data.split(":", 1)[1]
        if tag in selected:
            selected.remove(tag)
        else:
            selected.add(tag)
        context.user_data["selected_tags"] = selected
        await query.edit_message_reply_markup(reply_markup=build_keyboard(selected))

    elif data == "done":
        pending = context.user_data["pending_message"]
        text    = pending["text"]
        ts      = pending["ts"]
        tags    = list(selected)

        # Store in DB – off‑thread because psycopg2 is blocking
        await asyncio.to_thread(store_message, user.id, text, ts, tags)

        # Confirm save and clear user_data
        await query.edit_message_reply_markup(reply_markup=None)
        await query.edit_message_text("Saved! ✅")

        context.user_data.clear()

# ------------------------------------------------------------------
# Error handler (so the bot does not crash silently and we see the stack trace)
# ------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)

# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------

def main() -> None:
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    audio_filter = filters.VOICE | filters.AUDIO
    app.add_handler(MessageHandler(audio_filter, handle_audio))

    app.add_handler(CallbackQueryHandler(button_handler))

    app.add_error_handler(error_handler)

    print("Bot is running… press Ctrl‑C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()