"""
Discord Bot Module

Integrates the RAG Agent with Discord for the FAQ Chatbot.

Features:
- Responds to user questions with RAG-powered answers
- Maintains conversation context per user/channel
- Provides source citations
- Slash commands for help and status
- Rate limiting to prevent abuse

Design Rationale:
- Uses discord.py for Discord API integration
- Each channel gets its own conversation memory
- Confidence scores help users gauge answer reliability
- Sources provide transparency and trust

Usage:
    python -m src.discord_bot
    
    Or:
    from src.discord_bot import DiscordBot
    bot = DiscordBot()
    bot.run()
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
import discord
from discord import app_commands
from discord.ext import commands

from config.settings import get_settings, reload_settings
from src.rag_agent import RAGAgent

# Configure logging
logger = logging.getLogger(__name__)


class DiscordBot(commands.Bot):
    """
    Discord Bot with RAG-powered FAQ capabilities.
    
    This bot answers questions about the AI Bootcamp program
    using retrieval-augmented generation.
    
    Features:
    - Natural language Q&A
    - Conversation memory per channel
    - Source citations
    - Confidence scoring
    - Slash commands
    """
    
    def __init__(
        self,
        command_prefix: str = "!",
        rag_agent: Optional[RAGAgent] = None,
        **kwargs
    ):
        """
        Initialize the Discord Bot.
        
        Args:
            command_prefix: Prefix for text commands (default: "!")
            rag_agent: Optional pre-configured RAG agent
            **kwargs: Additional arguments for commands.Bot
        """
        # Set up intents - Only enable what we need
        # MESSAGE_CONTENT is required to read message content (must be enabled in Developer Portal)
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content - ENABLE IN DEVELOPER PORTAL
        intents.guilds = True
        # intents.members = True  # Disabled - not strictly needed, avoids extra privileged intent
        
        super().__init__(
            command_prefix=command_prefix,
            intents=intents,
            **kwargs
        )
        
        # RAG Agent - will be initialized on ready
        self._rag_agent = rag_agent
        self._is_ready = False
        
        # Rate limiting (simple in-memory)
        self._rate_limits: Dict[int, datetime] = {}
        self._rate_limit_seconds = 3  # Seconds between requests per user
        
        # Statistics
        self._stats = {
            "questions_answered": 0,
            "errors": 0,
            "start_time": None,
        }
        
        # Settings
        self.settings = get_settings()
        
        logger.info("DiscordBot initialized")
    
    @property
    def rag_agent(self) -> RAGAgent:
        """Get the RAG agent, initializing if needed."""
        if self._rag_agent is None:
            logger.info("Initializing RAG Agent...")
            self._rag_agent = RAGAgent(
                embedding_provider=self.settings.embedding.provider,
                llm_provider=self.settings.llm.provider,
                vector_store_provider=self.settings.vector_store.provider,
            )
            # Ingest knowledge base
            self._ingest_knowledge_base()
        return self._rag_agent
    
    def _ingest_knowledge_base(self):
        """Ingest the knowledge base documents."""
        kb_path = "./data/knowledge_base"
        if os.path.exists(kb_path):
            logger.info(f"Ingesting knowledge base from {kb_path}")
            stats = self.rag_agent.ingest_directory(kb_path)
            logger.info(f"Ingested: {stats}")
        else:
            logger.warning(f"Knowledge base path not found: {kb_path}")
    
    async def setup_hook(self):
        """Called when the bot is starting up."""
        # Register slash commands
        await self._register_commands()
        logger.info("Slash commands registered")
    
    async def _register_commands(self):
        """Register slash commands with Discord."""
        
        @self.tree.command(name="ask", description="Ask a question about the AI Bootcamp")
        @app_commands.describe(question="Your question about the AI Bootcamp program")
        async def ask_command(interaction: discord.Interaction, question: str):
            await self._handle_question(interaction, question)
        
        @self.tree.command(name="help", description="Get help using the FAQ bot")
        async def help_command(interaction: discord.Interaction):
            await self._send_help(interaction)
        
        @self.tree.command(name="status", description="Check bot status and statistics")
        async def status_command(interaction: discord.Interaction):
            await self._send_status(interaction)
        
        @self.tree.command(name="clear", description="Clear your conversation history")
        async def clear_command(interaction: discord.Interaction):
            await self._clear_history(interaction)
        
        # Sync commands with Discord
        try:
            synced = await self.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
    
    async def on_ready(self):
        """Called when the bot is ready and connected."""
        self._is_ready = True
        self._stats["start_time"] = datetime.now()
        
        logger.info(f"Bot is ready! Logged in as {self.user}")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")
        
        # Set bot status
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="your questions | /ask"
        )
        await self.change_presence(activity=activity)
        
        # Initialize RAG agent in background
        asyncio.create_task(self._async_init_rag())
    
    async def _async_init_rag(self):
        """Initialize RAG agent asynchronously."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.rag_agent)
            logger.info("RAG Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Ignore messages from other bots
        if message.author.bot:
            return
        
        # Check if the bot is mentioned or in DM
        is_mentioned = self.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Respond to mentions or DMs
        if is_mentioned or is_dm:
            # Remove bot mention from the message
            content = message.content
            if is_mentioned:
                content = content.replace(f"<@{self.user.id}>", "").strip()
                content = content.replace(f"<@!{self.user.id}>", "").strip()
            
            if content:
                await self._handle_message_question(message, content)
            else:
                # Just mentioned without a question
                await message.reply(
                    "👋 Hi! I'm the AI Bootcamp FAQ Bot. "
                    "Ask me a question about the bootcamp program!\n"
                    "Use `/ask` or just mention me with your question."
                )
        
        # Process text commands (if any)
        await self.process_commands(message)
    
    async def _handle_message_question(self, message: discord.Message, question: str):
        """Handle a question from a regular message."""
        # Check rate limit
        if not self._check_rate_limit(message.author.id):
            await message.reply(
                "⏳ Please wait a few seconds before asking another question.",
                delete_after=5
            )
            return
        
        # Show typing indicator
        async with message.channel.typing():
            try:
                # Get response from RAG agent
                response = await self._get_rag_response(
                    question=question,
                    conversation_id=str(message.channel.id),
                    user_id=str(message.author.id),
                )
                
                # Format and send response
                embed = self._format_response_embed(question, response)
                await message.reply(embed=embed)
                
                self._stats["questions_answered"] += 1
                
            except Exception as e:
                logger.error(f"Error handling question: {e}")
                self._stats["errors"] += 1
                await message.reply(
                    "❌ Sorry, I encountered an error processing your question. "
                    "Please try again later."
                )
    
    async def _handle_question(self, interaction: discord.Interaction, question: str):
        """Handle a question from a slash command."""
        # Check rate limit
        if not self._check_rate_limit(interaction.user.id):
            await interaction.response.send_message(
                "⏳ Please wait a few seconds before asking another question.",
                ephemeral=True
            )
            return
        
        # Defer the response (shows "thinking...")
        await interaction.response.defer()
        
        try:
            # Get response from RAG agent
            response = await self._get_rag_response(
                question=question,
                conversation_id=str(interaction.channel_id),
                user_id=str(interaction.user.id),
            )
            
            # Format and send response
            embed = self._format_response_embed(question, response)
            await interaction.followup.send(embed=embed)
            
            self._stats["questions_answered"] += 1
            
        except Exception as e:
            logger.error(f"Error handling slash command: {e}")
            self._stats["errors"] += 1
            await interaction.followup.send(
                "❌ Sorry, I encountered an error processing your question. "
                "Please try again later."
            )
    
    async def _get_rag_response(
        self,
        question: str,
        conversation_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get response from the RAG agent."""
        loop = asyncio.get_event_loop()
        
        # Run RAG query in executor (it's synchronous)
        response = await loop.run_in_executor(
            None,
            lambda: self.rag_agent.query(
                question=question,
                conversation_id=conversation_id,
            )
        )
        
        return response
    
    def _format_response_embed(
        self,
        question: str,
        response: Dict[str, Any]
    ) -> discord.Embed:
        """Format the RAG response as a Discord embed."""
        answer = response.get("answer", "No answer available.")
        confidence = response.get("confidence", 0)
        sources = response.get("sources", [])
        
        # Determine embed color based on confidence
        if confidence >= 0.6:
            color = discord.Color.green()
            confidence_emoji = "🟢"
        elif confidence >= 0.4:
            color = discord.Color.yellow()
            confidence_emoji = "🟡"
        else:
            color = discord.Color.orange()
            confidence_emoji = "🟠"
        
        # Create embed
        embed = discord.Embed(
            title="📚 AI Bootcamp FAQ",
            description=answer[:4000],  # Discord limit is 4096
            color=color,
            timestamp=datetime.now()
        )
        
        # Add question field
        embed.add_field(
            name="❓ Question",
            value=question[:1000],
            inline=False
        )
        
        # Add confidence
        embed.add_field(
            name=f"{confidence_emoji} Confidence",
            value=f"{confidence:.1%}",
            inline=True
        )
        
        # Add sources
        if sources:
            unique_sources = list(set(s.get("source", "Unknown") for s in sources[:3]))
            sources_text = "\n".join(f"• {s}" for s in unique_sources)
            embed.add_field(
                name="📄 Sources",
                value=sources_text[:1000],
                inline=True
            )
        
        # Add footer
        embed.set_footer(text="AI Bootcamp FAQ Bot | Use /help for more info")
        
        return embed
    
    async def _send_help(self, interaction: discord.Interaction):
        """Send help information."""
        embed = discord.Embed(
            title="🤖 AI Bootcamp FAQ Bot - Help",
            description="I can answer questions about the PM Accelerator AI Bootcamp program!",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="💬 How to Ask Questions",
            value=(
                "**Option 1:** Use `/ask` command\n"
                "**Option 2:** Mention me with your question\n"
                "**Option 3:** Send me a DM"
            ),
            inline=False
        )
        
        embed.add_field(
            name="📝 Example Questions",
            value=(
                "• How long is the bootcamp program?\n"
                "• What are the award tiers?\n"
                "• How does team matching work?\n"
                "• What happens during Week 1?\n"
                "• What is RAG?"
            ),
            inline=False
        )
        
        embed.add_field(
            name="⚡ Commands",
            value=(
                "`/ask` - Ask a question\n"
                "`/help` - Show this help message\n"
                "`/status` - Bot status and stats\n"
                "`/clear` - Clear conversation history"
            ),
            inline=False
        )
        
        embed.add_field(
            name="💡 Tips",
            value=(
                "• Be specific in your questions\n"
                "• Check the confidence score for reliability\n"
                "• Sources show where the answer came from"
            ),
            inline=False
        )
        
        embed.set_footer(text="Built with RAG (Retrieval-Augmented Generation)")
        
        await interaction.response.send_message(embed=embed)
    
    async def _send_status(self, interaction: discord.Interaction):
        """Send bot status information."""
        # Calculate uptime
        uptime = "N/A"
        if self._stats["start_time"]:
            delta = datetime.now() - self._stats["start_time"]
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{hours}h {minutes}m {seconds}s"
        
        # Get RAG stats
        rag_stats = self.rag_agent.get_stats()
        
        embed = discord.Embed(
            title="📊 Bot Status",
            color=discord.Color.green() if self._is_ready else discord.Color.red()
        )
        
        embed.add_field(
            name="🟢 Status",
            value="Online" if self._is_ready else "Initializing...",
            inline=True
        )
        
        embed.add_field(
            name="⏱️ Uptime",
            value=uptime,
            inline=True
        )
        
        embed.add_field(
            name="🏠 Servers",
            value=str(len(self.guilds)),
            inline=True
        )
        
        embed.add_field(
            name="❓ Questions Answered",
            value=str(self._stats["questions_answered"]),
            inline=True
        )
        
        embed.add_field(
            name="📚 Knowledge Base",
            value=f"{rag_stats.get('total_chunks', 0)} chunks",
            inline=True
        )
        
        embed.add_field(
            name="🤖 LLM Provider",
            value=self.settings.llm.provider.capitalize(),
            inline=True
        )
        
        embed.set_footer(text=f"Latency: {round(self.latency * 1000)}ms")
        
        await interaction.response.send_message(embed=embed)
    
    async def _clear_history(self, interaction: discord.Interaction):
        """Clear conversation history for the channel."""
        conversation_id = str(interaction.channel_id)
        
        try:
            self.rag_agent.clear_conversation(conversation_id)
            await interaction.response.send_message(
                "✅ Conversation history cleared!",
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            await interaction.response.send_message(
                "❌ Failed to clear conversation history.",
                ephemeral=True
            )
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is rate limited."""
        now = datetime.now()
        
        if user_id in self._rate_limits:
            last_request = self._rate_limits[user_id]
            elapsed = (now - last_request).total_seconds()
            
            if elapsed < self._rate_limit_seconds:
                return False
        
        self._rate_limits[user_id] = now
        return True
    
    def run_bot(self, token: Optional[str] = None):
        """
        Run the bot with the given token.
        
        Args:
            token: Discord bot token (or from environment)
        """
        token = token or os.getenv("DISCORD_BOT_TOKEN")
        
        if not token:
            raise ValueError(
                "Discord bot token not provided. "
                "Set DISCORD_BOT_TOKEN environment variable or pass token directly."
            )
        
        logger.info("Starting Discord bot...")
        self.run(token)


def create_bot(**kwargs) -> DiscordBot:
    """
    Factory function to create a configured Discord bot.
    
    Args:
        **kwargs: Arguments to pass to DiscordBot
        
    Returns:
        Configured DiscordBot instance
    """
    return DiscordBot(**kwargs)


# Main entry point
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("discord_bot.log"),
        ]
    )
    
    # Create and run bot
    bot = create_bot()
    bot.run_bot()
