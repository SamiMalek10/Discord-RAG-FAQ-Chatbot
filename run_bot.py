"""
Run Discord Bot - Direct launch script
"""
import sys
import logging
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

print("=" * 60)
print("  🤖 Discord RAG FAQ Chatbot - Starting...")
print("=" * 60)

from src.discord_bot import create_bot

print("""
Bot Commands:
  /ask <question>  - Ask a question about the AI Bootcamp
  /help            - Show help information  
  /status          - Show bot status
  /clear           - Clear conversation history

You can also mention the bot or DM it with your question.

Press Ctrl+C to stop the bot.
""")

# Create and run bot
bot = create_bot()
token = os.getenv("DISCORD_BOT_TOKEN")

if not token:
    print("❌ DISCORD_BOT_TOKEN not set in .env!")
    sys.exit(1)

# Remove quotes if present
token = token.strip('"').strip("'")

bot.run_bot(token)
