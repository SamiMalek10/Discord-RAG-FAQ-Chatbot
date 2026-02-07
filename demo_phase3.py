"""
Discord RAG FAQ Chatbot - Phase 3 Demo

This demo shows how to run the Discord bot with the RAG pipeline.

Prerequisites:
1. Create a Discord Application at https://discord.com/developers/applications
2. Create a Bot and get the token
3. Enable "Message Content Intent" in Bot settings
4. Invite the bot to your server with appropriate permissions
5. Set DISCORD_BOT_TOKEN in .env

Usage:
    python demo_phase3.py
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the demo banner."""
    print("=" * 60)
    print("  🤖 Discord RAG FAQ Chatbot - Phase 3 Demo")
    print("  Discord Bot Integration")
    print("=" * 60)


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("\n📋 Checking Prerequisites...\n")
    
    issues = []
    
    # Check Discord token
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if not discord_token or discord_token == "your_discord_bot_token_here":
        issues.append("❌ DISCORD_BOT_TOKEN not set in .env")
        print("   ❌ Discord Bot Token: NOT CONFIGURED")
    else:
        print("   ✅ Discord Bot Token: Configured")
    
    # Check LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"   ✅ LLM Provider: {llm_provider}")
    
    # Check knowledge base
    kb_path = "./data/knowledge_base"
    if os.path.exists(kb_path):
        files = [f for f in os.listdir(kb_path) if f.endswith(('.docx', '.pdf', '.txt'))]
        print(f"   ✅ Knowledge Base: {len(files)} documents")
    else:
        issues.append("❌ Knowledge base directory not found")
        print("   ❌ Knowledge Base: Not found")
    
    # Check discord.py
    try:
        import discord
        print(f"   ✅ discord.py: v{discord.__version__}")
    except ImportError:
        issues.append("❌ discord.py not installed")
        print("   ❌ discord.py: Not installed")
    
    return issues


def print_setup_instructions():
    """Print instructions for setting up the Discord bot."""
    print("\n" + "=" * 60)
    print("  📖 Discord Bot Setup Instructions")
    print("=" * 60)
    
    print("""
1. CREATE DISCORD APPLICATION
   - Go to: https://discord.com/developers/applications
   - Click "New Application"
   - Give it a name (e.g., "AI Bootcamp FAQ Bot")

2. CREATE BOT
   - Go to "Bot" section in left sidebar
   - Click "Add Bot"
   - Copy the TOKEN and save it

3. CONFIGURE BOT SETTINGS
   - In "Bot" section, enable these Intents:
     ✅ MESSAGE CONTENT INTENT (required to read messages)
     ✅ SERVER MEMBERS INTENT (optional)

4. SET ENVIRONMENT VARIABLE
   - Open your .env file
   - Set: DISCORD_BOT_TOKEN=your_token_here

5. INVITE BOT TO YOUR SERVER
   - Go to "OAuth2" > "URL Generator"
   - Select scopes:
     ✅ bot
     ✅ applications.commands
   - Select bot permissions:
     ✅ Send Messages
     ✅ Embed Links
     ✅ Read Message History
     ✅ Use Slash Commands
   - Copy the generated URL and open it in browser
   - Select your server and authorize

6. RUN THE BOT
   - python demo_phase3.py
   - Or: python -m src.discord_bot
""")


def run_bot():
    """Run the Discord bot."""
    from src.discord_bot import create_bot
    
    print("\n" + "=" * 60)
    print("  🚀 Starting Discord Bot...")
    print("=" * 60)
    
    print("""
Bot Commands:
  /ask <question>  - Ask a question about the AI Bootcamp
  /help            - Show help information
  /status          - Show bot status and statistics
  /clear           - Clear conversation history

You can also:
  - Mention the bot with your question
  - Send a DM to the bot

Press Ctrl+C to stop the bot.
""")
    
    # Create and run the bot
    bot = create_bot()
    
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token or token == "your_discord_bot_token_here":
        print("❌ Error: DISCORD_BOT_TOKEN not set!")
        print("   Please set it in your .env file.")
        return
    
    try:
        bot.run_bot(token)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running bot: {e}")
        logger.exception("Bot error")


def run_test_mode():
    """Run in test mode without actually connecting to Discord."""
    print("\n" + "=" * 60)
    print("  🧪 Running in Test Mode")
    print("=" * 60)
    
    print("\nTesting RAG Agent integration...\n")
    
    from config.settings import reload_settings
    from src.rag_agent import RAGAgent
    
    # Reload settings
    reload_settings()
    
    # Initialize RAG Agent
    print("📦 Initializing RAG Agent...")
    agent = RAGAgent(
        embedding_provider='local',
        llm_provider='mistral',
        vector_store_provider='faiss'
    )
    
    # Ingest documents
    print("📥 Ingesting knowledge base...")
    stats = agent.ingest_directory('./data/knowledge_base')
    print(f"   ✅ Ingested {stats.get('total_documents', 0)} documents")
    
    # Test queries that the Discord bot would receive
    test_questions = [
        "How long is the bootcamp?",
        "What are the award tiers?",
        "How does team matching work?",
    ]
    
    print("\n📝 Simulating Discord Bot Queries:\n")
    print("-" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Query {i}] {question}")
        
        response = agent.query(question, conversation_id="test_channel_123")
        
        answer = response['answer']
        confidence = response['confidence']
        sources = response['sources']
        
        # Simulate Discord embed formatting
        if confidence >= 0.6:
            confidence_emoji = "🟢"
        elif confidence >= 0.4:
            confidence_emoji = "🟡"
        else:
            confidence_emoji = "🟠"
        
        print(f"\n{confidence_emoji} Confidence: {confidence:.1%}")
        print(f"📝 Answer: {answer[:300]}...")
        
        if sources:
            unique_sources = list(set(s.get("source", "Unknown") for s in sources[:3]))
            print(f"📄 Sources: {', '.join(unique_sources)}")
        
        print("-" * 50)
    
    print("\n✅ Test mode complete!")
    print("   The RAG Agent is working correctly.")
    print("   Set DISCORD_BOT_TOKEN to run the actual bot.")


def main():
    """Main entry point."""
    print_banner()
    
    issues = check_prerequisites()
    
    if issues:
        print("\n⚠️  Some prerequisites are missing:")
        for issue in issues:
            print(f"   {issue}")
        
        if "DISCORD_BOT_TOKEN" in str(issues):
            print_setup_instructions()
            
            # Offer test mode
            print("\n" + "=" * 60)
            choice = input("\nWould you like to run in test mode instead? (y/n): ").strip().lower()
            if choice == 'y':
                run_test_mode()
        return
    
    print("\n✅ All prerequisites met!")
    
    # Ask user what to do
    print("\nOptions:")
    print("  1. Run the Discord bot")
    print("  2. Run in test mode (no Discord connection)")
    print("  3. Show setup instructions")
    print("  4. Exit")
    
    choice = input("\nSelect an option (1-4): ").strip()
    
    if choice == '1':
        run_bot()
    elif choice == '2':
        run_test_mode()
    elif choice == '3':
        print_setup_instructions()
    else:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
