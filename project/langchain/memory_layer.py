import sys
import pathlib
import time
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

# Ensure roots are in path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import chat_memory
from context_pipeline.config import PipelineConfig
from context_pipeline.memory_decay import trim_chat_messages
from context_pipeline.token_budget import count_tokens

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by our existing sqlite3 chat_memory database."""

    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        db_msgs = chat_memory.get_messages(self.session_id)
        langchain_msgs = []
        for msg in db_msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                langchain_msgs.append(AIMessage(content=content))
            elif role == "user":
                langchain_msgs.append(HumanMessage(content=content))
            elif role == "system":
                langchain_msgs.append(SystemMessage(content=content))
            else:
                langchain_msgs.append(HumanMessage(content=content))
        return langchain_msgs

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "user"
        chat_memory.save_message(self.session_id, role, message.content)

    def add_user_message(self, content: str) -> None:
        chat_memory.save_message(self.session_id, "user", content)

    def add_ai_message(self, content: str) -> None:
        chat_memory.save_message(self.session_id, "assistant", content)

    def clear(self) -> None:
        chat_memory.clear_messages(self.session_id)

    def get_decayed_messages(self, model: str, max_tokens: int) -> List[BaseMessage]:
        """Loads messages from the DB, runs them through the memory_decay module, and returns the trimmed list."""
        db_msgs = chat_memory.get_messages(self.session_id)
        if not db_msgs:
            return []
            
        cfg = PipelineConfig.from_env()
        # Trim history
        trimmed_db_msgs = trim_chat_messages(
            db_msgs,
            token_counter=lambda t: count_tokens(t, model),
            max_tokens=max_tokens,
            cfg=cfg
        )
        
        # Convert to LangChain messages
        langchain_msgs = []
        for msg in trimmed_db_msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                langchain_msgs.append(AIMessage(content=content))
            elif role == "user":
                langchain_msgs.append(HumanMessage(content=content))
            elif role == "system":
                langchain_msgs.append(SystemMessage(content=content))
            else:
                langchain_msgs.append(HumanMessage(content=content))
        return langchain_msgs
