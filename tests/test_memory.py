"""
Tests for Conversation Memory Module

Tests for Message, ConversationMemory, and ConversationManager.
"""

import pytest
from datetime import datetime, timedelta
from src.memory import Message, ConversationMemory, ConversationManager


class TestMessage:
    """Tests for the Message dataclass."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role="assistant",
            content="Response",
            metadata={"sources": ["doc1.pdf"]}
        )
        assert msg.metadata["sources"] == ["doc1.pdf"]
    
    def test_message_to_dict(self):
        """Test serialization to dictionary."""
        msg = Message(role="user", content="Test")
        d = msg.to_dict()
        
        assert d["role"] == "user"
        assert d["content"] == "Test"
        assert "timestamp" in d
        assert "metadata" in d
    
    def test_message_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "role": "assistant",
            "content": "Hello!",
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"key": "value"}
        }
        msg = Message.from_dict(data)
        
        assert msg.role == "assistant"
        assert msg.content == "Hello!"
        assert msg.metadata["key"] == "value"
    
    def test_message_str(self):
        """Test string representation."""
        msg = Message(role="user", content="Hi there")
        assert str(msg) == "user: Hi there"


class TestConversationMemory:
    """Tests for ConversationMemory class."""
    
    def test_initialization(self):
        """Test memory initialization."""
        memory = ConversationMemory()
        assert len(memory) == 0
        assert memory.conversation_id is not None
    
    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        memory = ConversationMemory(
            max_turns=5,
            max_tokens=1000,
            conversation_id="test_123"
        )
        assert memory.max_turns == 5
        assert memory.max_tokens == 1000
        assert memory.conversation_id == "test_123"
    
    def test_add_user_message(self):
        """Test adding user message."""
        memory = ConversationMemory()
        memory.add_user_message("Hello")
        
        assert len(memory) == 1
        messages = memory.get_messages()
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
    
    def test_add_assistant_message(self):
        """Test adding assistant message."""
        memory = ConversationMemory()
        memory.add_assistant_message("Hi there!")
        
        assert len(memory) == 1
        messages = memory.get_messages()
        assert messages[0].role == "assistant"
    
    def test_conversation_flow(self):
        """Test multi-turn conversation."""
        memory = ConversationMemory()
        
        memory.add_user_message("What is AI?")
        memory.add_assistant_message("AI is artificial intelligence.")
        memory.add_user_message("Tell me more.")
        memory.add_assistant_message("It involves machine learning...")
        
        assert len(memory) == 4
    
    def test_get_messages_for_llm(self):
        """Test getting messages in LLM format."""
        memory = ConversationMemory()
        memory.set_system_prompt("You are helpful.")
        memory.add_user_message("Hi")
        memory.add_assistant_message("Hello!")
        
        messages = memory.get_messages_for_llm()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_get_context_string(self):
        """Test getting conversation as string."""
        memory = ConversationMemory()
        memory.add_user_message("Question?")
        memory.add_assistant_message("Answer!")
        
        context = memory.get_context_string()
        
        assert "User: Question?" in context
        assert "Assistant: Answer!" in context
    
    def test_get_last_user_message(self):
        """Test getting last user message."""
        memory = ConversationMemory()
        memory.add_user_message("First")
        memory.add_assistant_message("Response")
        memory.add_user_message("Second")
        
        assert memory.get_last_user_message() == "Second"
    
    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        memory = ConversationMemory()
        memory.add_user_message("Question")
        memory.add_assistant_message("First answer")
        memory.add_user_message("Follow up")
        memory.add_assistant_message("Second answer")
        
        assert memory.get_last_assistant_message() == "Second answer"
    
    def test_clear(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        memory.add_user_message("Test")
        memory.add_assistant_message("Response")
        
        memory.clear()
        
        assert len(memory) == 0
    
    def test_max_turns_limit(self):
        """Test that max turns is enforced."""
        memory = ConversationMemory(max_turns=2)
        
        # Add more than max turns
        for i in range(5):
            memory.add_user_message(f"User message {i}")
            memory.add_assistant_message(f"Assistant message {i}")
        
        # Should only keep last 4 messages (2 turns * 2 messages)
        assert len(memory) <= 4
    
    def test_token_limit_trimming(self):
        """Test that token limit trims old messages."""
        memory = ConversationMemory(max_tokens=100)
        
        # Add long messages
        memory.add_user_message("A" * 200)  # Way over limit
        memory.add_assistant_message("B" * 200)
        memory.add_user_message("C" * 50)
        
        # Should have trimmed old messages
        messages = memory.get_messages()
        total_chars = sum(len(m.content) for m in messages)
        assert total_chars < 500  # Should be trimmed
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        memory = ConversationMemory(conversation_id="test")
        memory.set_system_prompt("System")
        memory.add_user_message("Hello")
        
        d = memory.to_dict()
        
        assert d["conversation_id"] == "test"
        assert d["system_prompt"] == "System"
        assert len(d["messages"]) == 1
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "conversation_id": "restored",
            "system_prompt": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hi", "metadata": {}}
            ],
            "max_turns": 10,
            "max_tokens": 2000,
        }
        
        memory = ConversationMemory.from_dict(data)
        
        assert memory.conversation_id == "restored"
        assert len(memory) == 1


class TestConversationManager:
    """Tests for ConversationManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = ConversationManager()
        assert len(manager) == 0
    
    def test_get_memory_creates_new(self):
        """Test that get_memory creates new memory."""
        manager = ConversationManager()
        
        memory = manager.get_memory("user_1")
        
        assert memory is not None
        assert len(manager) == 1
    
    def test_get_memory_returns_existing(self):
        """Test that get_memory returns existing memory."""
        manager = ConversationManager()
        
        memory1 = manager.get_memory("user_1")
        memory1.add_user_message("Hello")
        
        memory2 = manager.get_memory("user_1")
        
        assert memory2 is memory1
        assert len(memory2) == 1
    
    def test_get_memory_no_create(self):
        """Test get_memory with create_if_missing=False."""
        manager = ConversationManager()
        
        memory = manager.get_memory("nonexistent", create_if_missing=False)
        
        assert memory is None
    
    def test_delete_memory(self):
        """Test deleting memory."""
        manager = ConversationManager()
        manager.get_memory("user_1")
        
        assert manager.delete_memory("user_1") is True
        assert len(manager) == 0
    
    def test_delete_nonexistent(self):
        """Test deleting nonexistent memory."""
        manager = ConversationManager()
        
        assert manager.delete_memory("nonexistent") is False
    
    def test_clear_all(self):
        """Test clearing all memories."""
        manager = ConversationManager()
        manager.get_memory("user_1")
        manager.get_memory("user_2")
        manager.get_memory("user_3")
        
        count = manager.clear_all()
        
        assert count == 3
        assert len(manager) == 0
    
    def test_get_all_conversation_ids(self):
        """Test getting all conversation IDs."""
        manager = ConversationManager()
        manager.get_memory("alice")
        manager.get_memory("bob")
        
        ids = manager.get_all_conversation_ids()
        
        assert "alice" in ids
        assert "bob" in ids
    
    def test_multiple_users(self):
        """Test managing multiple user conversations."""
        manager = ConversationManager()
        
        # User 1 conversation
        m1 = manager.get_memory("user_1")
        m1.add_user_message("Hello from user 1")
        
        # User 2 conversation
        m2 = manager.get_memory("user_2")
        m2.add_user_message("Hello from user 2")
        
        # Verify isolation
        assert manager.get_memory("user_1").get_last_user_message() == "Hello from user 1"
        assert manager.get_memory("user_2").get_last_user_message() == "Hello from user 2"
