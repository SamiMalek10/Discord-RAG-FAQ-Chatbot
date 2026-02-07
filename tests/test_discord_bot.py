"""
Tests for Discord Bot Module

Tests the Discord bot integration with the RAG Agent.
Uses mocking to avoid actual Discord API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import discord

# Import the bot module
from src.discord_bot import DiscordBot, create_bot


class TestDiscordBot:
    """Tests for DiscordBot class."""
    
    def test_initialization(self):
        """Test bot initialization."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            assert bot._rag_agent is None  # Lazy initialization
            assert bot._is_ready is False
            assert bot._rate_limit_seconds == 3
            assert bot._stats["questions_answered"] == 0
    
    def test_initialization_with_prefix(self):
        """Test bot with custom command prefix."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot(command_prefix="?")
            
            assert bot.command_prefix == "?"
    
    def test_initialization_with_rag_agent(self):
        """Test bot with pre-configured RAG agent."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            mock_agent = Mock()
            bot = DiscordBot(rag_agent=mock_agent)
            
            assert bot._rag_agent is mock_agent
    
    def test_rate_limit_check_first_request(self):
        """Test rate limiting allows first request."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # First request should be allowed
            assert bot._check_rate_limit(123456) is True
    
    def test_rate_limit_blocks_rapid_requests(self):
        """Test rate limiting blocks rapid requests."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # First request
            assert bot._check_rate_limit(123456) is True
            
            # Immediate second request should be blocked
            assert bot._check_rate_limit(123456) is False
    
    def test_rate_limit_different_users(self):
        """Test rate limiting is per-user."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # First user
            assert bot._check_rate_limit(123456) is True
            
            # Different user should not be blocked
            assert bot._check_rate_limit(789012) is True
    
    def test_format_response_embed_high_confidence(self):
        """Test embed formatting with high confidence."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            response = {
                "answer": "The bootcamp is 3 weeks long.",
                "confidence": 0.75,
                "sources": [{"source": "FAQ.docx"}],
            }
            
            embed = bot._format_response_embed("How long is bootcamp?", response)
            
            assert isinstance(embed, discord.Embed)
            assert embed.color == discord.Color.green()
            assert "3 weeks" in embed.description
    
    def test_format_response_embed_medium_confidence(self):
        """Test embed formatting with medium confidence."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            response = {
                "answer": "The bootcamp covers AI topics.",
                "confidence": 0.5,
                "sources": [],
            }
            
            embed = bot._format_response_embed("What is covered?", response)
            
            assert embed.color == discord.Color.yellow()
    
    def test_format_response_embed_low_confidence(self):
        """Test embed formatting with low confidence."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            response = {
                "answer": "I'm not sure about that.",
                "confidence": 0.2,
                "sources": [],
            }
            
            embed = bot._format_response_embed("Random question?", response)
            
            assert embed.color == discord.Color.orange()
    
    def test_format_response_embed_with_sources(self):
        """Test embed includes sources."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            response = {
                "answer": "The award tiers are...",
                "confidence": 0.6,
                "sources": [
                    {"source": "FAQ.docx"},
                    {"source": "Training.docx"},
                ],
            }
            
            embed = bot._format_response_embed("What are award tiers?", response)
            
            # Check sources are in the embed
            sources_field = None
            for field in embed.fields:
                if "Sources" in field.name:
                    sources_field = field
                    break
            
            assert sources_field is not None
            assert "FAQ.docx" in sources_field.value
    
    def test_format_response_truncates_long_answer(self):
        """Test that long answers are truncated."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Create a very long answer
            long_answer = "A" * 5000
            
            response = {
                "answer": long_answer,
                "confidence": 0.5,
                "sources": [],
            }
            
            embed = bot._format_response_embed("Question?", response)
            
            # Discord limit is 4096, but we set 4000
            assert len(embed.description) <= 4000


class TestCreateBot:
    """Tests for the create_bot factory function."""
    
    def test_create_bot_default(self):
        """Test creating bot with defaults."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = create_bot()
            
            assert isinstance(bot, DiscordBot)
    
    def test_create_bot_with_kwargs(self):
        """Test creating bot with custom arguments."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = create_bot(command_prefix=">>")
            
            assert bot.command_prefix == ">>"


class TestBotEvents:
    """Tests for bot event handlers."""
    
    @pytest.mark.asyncio
    async def test_on_ready_sets_status(self):
        """Test on_ready event sets bot status."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Use object.__setattr__ to bypass property restrictions for testing
            object.__setattr__(bot, '_user', Mock(id=123456))
            bot._guilds = [Mock()]
            bot.change_presence = AsyncMock()
            
            # Mock the RAG agent initialization
            with patch.object(bot, '_async_init_rag', new_callable=AsyncMock):
                # Also patch the user property access
                with patch.object(type(bot), 'user', new_callable=lambda: property(lambda self: Mock(id=123456))):
                    with patch.object(type(bot), 'guilds', new_callable=lambda: property(lambda self: [Mock()])):
                        await bot.on_ready()
            
            assert bot._is_ready is True
            assert bot._stats["start_time"] is not None
            bot.change_presence.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_question_updates_stats(self):
        """Test that handling a question updates statistics."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Mock the RAG agent
            mock_agent = Mock()
            mock_agent.query.return_value = {
                "answer": "Test answer",
                "confidence": 0.8,
                "sources": [],
            }
            bot._rag_agent = mock_agent
            
            # Mock interaction
            interaction = AsyncMock()
            interaction.user = Mock(id=123456)
            interaction.channel_id = 789
            
            # Reset rate limit
            bot._rate_limits = {}
            
            await bot._handle_question(interaction, "Test question?")
            
            assert bot._stats["questions_answered"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_question_rate_limited(self):
        """Test that rate limited requests are rejected."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Pre-populate rate limit
            bot._rate_limits[123456] = datetime.now()
            
            # Mock interaction
            interaction = AsyncMock()
            interaction.user = Mock(id=123456)
            interaction.response = AsyncMock()
            
            await bot._handle_question(interaction, "Test question?")
            
            # Should send ephemeral rate limit message
            interaction.response.send_message.assert_called_once()
            call_args = interaction.response.send_message.call_args
            assert call_args[1].get("ephemeral") is True


class TestBotCommands:
    """Tests for bot slash commands behavior."""
    
    @pytest.mark.asyncio
    async def test_clear_history_success(self):
        """Test clearing conversation history."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Mock RAG agent
            mock_agent = Mock()
            mock_agent.clear_conversation = Mock()
            bot._rag_agent = mock_agent
            
            # Mock interaction
            interaction = AsyncMock()
            interaction.channel_id = 123456
            interaction.response = AsyncMock()
            
            await bot._clear_history(interaction)
            
            mock_agent.clear_conversation.assert_called_once_with("123456")
            interaction.response.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_help(self):
        """Test help command sends embed."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            
            # Mock interaction
            interaction = AsyncMock()
            interaction.response = AsyncMock()
            
            await bot._send_help(interaction)
            
            interaction.response.send_message.assert_called_once()
            call_args = interaction.response.send_message.call_args
            embed = call_args[1].get("embed")
            
            assert embed is not None
            assert "Help" in embed.title
    
    @pytest.mark.asyncio
    async def test_send_status(self):
        """Test status command sends embed."""
        with patch('src.discord_bot.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                embedding=Mock(provider='local'),
                llm=Mock(provider='mistral'),
                vector_store=Mock(provider='faiss'),
            )
            
            bot = DiscordBot()
            bot._stats["start_time"] = datetime.now()
            bot._is_ready = True
            
            # Mock RAG agent
            mock_agent = Mock()
            mock_agent.get_stats.return_value = {"total_chunks": 100}
            bot._rag_agent = mock_agent
            
            # Mock interaction
            interaction = AsyncMock()
            interaction.response = AsyncMock()
            
            # Patch read-only properties
            with patch.object(type(bot), 'guilds', new_callable=lambda: property(lambda self: [Mock(), Mock()])):
                with patch.object(type(bot), 'latency', new_callable=lambda: property(lambda self: 0.05)):
                    await bot._send_status(interaction)
            
            interaction.response.send_message.assert_called_once()
            call_args = interaction.response.send_message.call_args
            embed = call_args[1].get("embed")
            
            assert embed is not None
            assert "Status" in embed.title
