"""
Conversation Memory Module

Manages conversation history and context for multi-turn interactions.
Implements a sliding window approach to manage context length.

Design Rationale:
- Preserves conversation context for follow-up questions
- Manages token limits by truncating old messages
- Thread-safe for concurrent Discord bot usage
- No global variables (per assignment requirement)

Usage:
    memory = ConversationMemory(max_turns=10)
    memory.add_user_message("What is the bootcamp schedule?")
    memory.add_assistant_message("The bootcamp runs for 11 weeks...")
    
    context = memory.get_context_string()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import deque
import threading
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Represents a single message in the conversation.
    
    Attributes:
        role: "user", "assistant", or "system"
        content: The message text
        timestamp: When the message was created
        metadata: Additional info (sources, confidence, etc.)
    """
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {}),
        )
    
    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class ConversationMemory:
    """
    Manages conversation history for a single conversation thread.
    
    Features:
    - Sliding window to limit context size
    - Automatic truncation of old messages
    - Token estimation for context management
    - Export/import for persistence
    
    Example:
        memory = ConversationMemory(max_turns=10)
        
        # Add messages
        memory.add_user_message("What is RAG?")
        memory.add_assistant_message("RAG stands for Retrieval-Augmented Generation...")
        
        # Get formatted context for LLM
        context = memory.get_context_string()
        
        # Get as list of dicts (for OpenAI API format)
        messages = memory.get_messages_for_llm()
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        max_tokens: int = 2000,
        conversation_id: Optional[str] = None,
    ):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
            max_tokens: Approximate max tokens for context (uses char estimate)
            conversation_id: Unique ID for this conversation
        """
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.conversation_id = conversation_id or self._generate_id()
        
        # Use deque for efficient FIFO operations
        self._messages: deque[Message] = deque(maxlen=max_turns * 2)
        self._system_prompt: Optional[str] = None
        self._lock = threading.Lock()
        
        logger.debug(f"ConversationMemory created: id={self.conversation_id}")
    
    def _generate_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt for this conversation.
        
        Args:
            prompt: System instructions for the LLM
        """
        with self._lock:
            self._system_prompt = prompt
    
    def add_user_message(
        self, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            content: User's message text
            metadata: Optional metadata
        """
        message = Message(
            role="user",
            content=content,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._messages.append(message)
            self._trim_to_token_limit()
        
        logger.debug(f"Added user message to {self.conversation_id}")
    
    def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Assistant's response text
            metadata: Optional metadata (sources, confidence, etc.)
        """
        message = Message(
            role="assistant",
            content=content,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._messages.append(message)
            self._trim_to_token_limit()
        
        logger.debug(f"Added assistant message to {self.conversation_id}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Uses the approximation: 1 token ≈ 4 characters for English.
        """
        return len(text) // 4
    
    def _trim_to_token_limit(self) -> None:
        """Trim old messages to stay within token limit."""
        total_tokens = sum(
            self._estimate_tokens(m.content) for m in self._messages
        )
        
        if self._system_prompt:
            total_tokens += self._estimate_tokens(self._system_prompt)
        
        # Remove oldest messages until under limit
        while total_tokens > self.max_tokens and len(self._messages) > 2:
            removed = self._messages.popleft()
            total_tokens -= self._estimate_tokens(removed.content)
    
    def get_messages(self) -> List[Message]:
        """
        Get all messages in the conversation.
        
        Returns:
            List of Message objects
        """
        with self._lock:
            return list(self._messages)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get messages in LLM API format.
        
        Returns format compatible with OpenAI/Ollama chat APIs:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            List of message dicts
        """
        with self._lock:
            messages = []
            
            # Add system prompt if set
            if self._system_prompt:
                messages.append({
                    "role": "system",
                    "content": self._system_prompt,
                })
            
            # Add conversation messages
            for msg in self._messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
            
            return messages
    
    def get_context_string(self) -> str:
        """
        Get conversation as a formatted string.
        
        Useful for including in prompts.
        
        Returns:
            Formatted conversation string
        """
        with self._lock:
            lines = []
            
            if self._system_prompt:
                lines.append(f"System: {self._system_prompt}")
                lines.append("")
            
            for msg in self._messages:
                role = msg.role.capitalize()
                lines.append(f"{role}: {msg.content}")
            
            return "\n".join(lines)
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == "user":
                    return msg.content
            return None
    
    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message."""
        with self._lock:
            for msg in reversed(self._messages):
                if msg.role == "assistant":
                    return msg.content
            return None
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        with self._lock:
            self._messages.clear()
        logger.debug(f"Cleared memory for {self.conversation_id}")
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export memory to dictionary for persistence."""
        with self._lock:
            return {
                "conversation_id": self.conversation_id,
                "system_prompt": self._system_prompt,
                "messages": [m.to_dict() for m in self._messages],
                "max_turns": self.max_turns,
                "max_tokens": self.max_tokens,
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Create memory from dictionary."""
        memory = cls(
            max_turns=data.get("max_turns", 10),
            max_tokens=data.get("max_tokens", 2000),
            conversation_id=data.get("conversation_id"),
        )
        
        if data.get("system_prompt"):
            memory.set_system_prompt(data["system_prompt"])
        
        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            memory._messages.append(msg)
        
        return memory


class ConversationManager:
    """
    Manages multiple conversation memories (for multi-user scenarios).
    
    Thread-safe manager for handling multiple concurrent conversations,
    such as in a Discord bot with multiple users.
    
    Example:
        manager = ConversationManager()
        
        # Get or create memory for a user/channel
        memory = manager.get_memory("user_123")
        memory.add_user_message("Hello!")
        
        # Clean up old conversations
        manager.cleanup_old_conversations(max_age_hours=24)
    """
    
    def __init__(self, default_max_turns: int = 10):
        """
        Initialize conversation manager.
        
        Args:
            default_max_turns: Default max turns for new conversations
        """
        self._memories: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self.default_max_turns = default_max_turns
        
        logger.info("ConversationManager initialized")
    
    def get_memory(
        self, 
        conversation_id: str,
        create_if_missing: bool = True,
    ) -> Optional[ConversationMemory]:
        """
        Get memory for a conversation ID.
        
        Args:
            conversation_id: Unique ID (e.g., user_id, channel_id)
            create_if_missing: Create new memory if not found
            
        Returns:
            ConversationMemory instance or None
        """
        with self._lock:
            if conversation_id not in self._memories:
                if create_if_missing:
                    self._memories[conversation_id] = ConversationMemory(
                        max_turns=self.default_max_turns,
                        conversation_id=conversation_id,
                    )
                    logger.debug(f"Created new memory for {conversation_id}")
                else:
                    return None
            
            return self._memories[conversation_id]
    
    def delete_memory(self, conversation_id: str) -> bool:
        """
        Delete a conversation memory.
        
        Args:
            conversation_id: ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if conversation_id in self._memories:
                del self._memories[conversation_id]
                logger.debug(f"Deleted memory for {conversation_id}")
                return True
            return False
    
    def clear_all(self) -> int:
        """
        Clear all conversation memories.
        
        Returns:
            Number of memories cleared
        """
        with self._lock:
            count = len(self._memories)
            self._memories.clear()
            logger.info(f"Cleared {count} conversation memories")
            return count
    
    def cleanup_old_conversations(self, max_age_hours: float = 24) -> int:
        """
        Remove conversations with no recent activity.
        
        Args:
            max_age_hours: Max hours since last message
            
        Returns:
            Number of conversations removed
        """
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed = 0
        
        with self._lock:
            to_remove = []
            
            for conv_id, memory in self._memories.items():
                messages = memory.get_messages()
                if messages:
                    last_msg_time = messages[-1].timestamp
                    if last_msg_time < cutoff:
                        to_remove.append(conv_id)
                else:
                    to_remove.append(conv_id)
            
            for conv_id in to_remove:
                del self._memories[conv_id]
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old conversations")
        
        return removed
    
    def get_all_conversation_ids(self) -> List[str]:
        """Get all active conversation IDs."""
        with self._lock:
            return list(self._memories.keys())
    
    def __len__(self) -> int:
        """Return number of active conversations."""
        return len(self._memories)
