import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


@dataclass
class SessionData:
    session_id: str
    created_at: float
    last_accessed: float
    analysis_cache: Dict[str, Any]
    metadata: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]


class LangChainRedisSessionManager:
    """Redis-based session manager with LangChain integration for BaseMessage."""

    def __init__(self, default_ttl: int = None):
        self.default_ttl = default_ttl or getattr(
            settings.redis, "default_ttl", 3600
        )
        self.redis = None
        self._lock = asyncio.Lock()
        logger.info(
            f"LangChainRedisSessionManager initialized - TTL: {self.default_ttl}s"
        )

    def start(self):
        """Initialize Redis connection to ElastiCache with password-based auth."""
        try:
            logger.info("Initializing Redis connection to ElastiCache...")

            # Lấy settings từ environment variables
            redis_host = os.getenv("REDIS_HOST", settings.redis.host)
            redis_port = int(
                os.getenv("REDIS_PORT", getattr(settings.redis, "port", 6379))
            )
            redis_db = int(
                os.getenv("REDIS_DB", 0)
            )  # Cố định DB 0 cho ElastiCache
            redis_password = os.getenv("REDIS_PASSWORD")  # Từ Parameter Store
            pool_size = int(
                os.getenv(
                    "REDIS_POOL_SIZE", getattr(settings.redis, "pool_size", 20)
                )
            )
            socket_timeout = float(
                os.getenv(
                    "REDIS_SOCKET_TIMEOUT",
                    getattr(settings.redis, "socket_timeout", 5),
                )
            )
            connection_timeout = float(
                os.getenv(
                    "REDIS_CONNECTION_TIMEOUT",
                    getattr(settings.redis, "connection_timeout", 5),
                )
            )
            health_check_interval = int(
                os.getenv("REDIS_HEALTH_CHECK_INTERVAL", 30)
            )

            self.redis = redis.Redis(
                host=redis_host,  # ElastiCache endpoint: ai-agent-redis-cluster.xxxxx.0001.use1.cache.amazonaws.com
                port=redis_port,
                db=redis_db,
                password=redis_password,  # Auth token từ Parameter Store
                decode_responses=True,
                # ssl=True,  # TLS required
                # ssl_cert_reqs=None,  # Bỏ qua SSL verification (production: dùng CA cert)
                max_connections=pool_size,
                socket_timeout=socket_timeout,
                socket_connect_timeout=connection_timeout,
                health_check_interval=health_check_interval,
                retry_on_timeout=True,
                # socket_keepalive=True,
                # retry_on_error=[ConnectionError],
            )

            # Test connection
            self.redis.ping()
            logger.info(
                "Redis connection to ElastiCache established successfully"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise

    async def stop(self):
        """Stop Redis connection."""
        try:
            logger.info("Stopping Redis session manager...")
            if self.redis:
                await self.redis.close()
                logger.info("Redis connection closed")
            logger.info("Redis session manager stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop Redis session manager: {str(e)}")
            raise

    async def create_session(
        self, metadata: Optional[Dict[str, Any]] = {}
    ) -> str:
        """Create a new session with a unique ID."""
        async with self._lock:
            session_id = str(uuid.uuid4())
            session_data = SessionData(
                session_id=session_id,
                created_at=time.time(),
                last_accessed=time.time(),
                analysis_cache={},
                metadata=metadata,
                conversation_history=[],
            )
            await self.redis.setex(
                self._get_session_key(session_id),
                self.default_ttl,
                json.dumps(self._serialize_session_data(session_data)),
            )
            return session_id

    async def create_session_with_id(
        self, session_id: str, metadata: Optional[Dict[str, Any]] = {}
    ):
        """Create a session with a specific ID, checking if it already exists."""
        async with self._lock:
            if not session_id or not session_id.strip():
                logger.error(
                    "Attempted to create session with empty session_id"
                )
                raise ValueError("Session ID cannot be empty")

            session_key = self._get_session_key(session_id)
            exists = await self.redis.exists(session_key)
            if exists:
                logger.error(f"Session with ID {session_id} already exists")
                raise ValueError(
                    f"Session with ID {session_id} already exists"
                )

            session_data = SessionData(
                session_id=session_id,
                created_at=time.time(),
                last_accessed=time.time(),
                analysis_cache={},
                metadata=metadata,
                conversation_history=[],
            )
            await self.redis.setex(
                session_key,
                self.default_ttl,
                json.dumps(self._serialize_session_data(session_data)),
            )
            logger.info(f"Created session with ID: {session_id}")

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by ID."""
        try:
            session_key = self._get_session_key(session_id)
            data = await self.redis.get(session_key)
            if data:
                session_data = self._deserialize_session_data(json.loads(data))
                session_data.last_accessed = time.time()
                await self.redis.setex(
                    session_key,
                    self.default_ttl,
                    json.dumps(self._serialize_session_data(session_data)),
                )
                return session_data
            return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None

    async def update_conversation(self, session_id: str, message: BaseMessage):
        """Update conversation history using RedisChatMessageHistory."""
        try:
            history = RedisChatMessageHistory(
                session_id=session_id,
                url=f"redis://{settings.redis.username}:{settings.redis.password}@{settings.redis.host}:{settings.redis.port}/{settings.redis.db}",
                ttl=self.default_ttl,
            )
            if isinstance(message, HumanMessage):
                history.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                history.add_ai_message(message.content)
            elif isinstance(message, SystemMessage):
                history.add_message(message)
            logger.info(f"Updated conversation for session: {session_id}")

            # Update session data with conversation history
            session_data = await self.get_session(session_id)
            if session_data:
                session_data.conversation_history = [
                    {"type": msg.__class__.__name__, "content": msg.content}
                    for msg in history.messages
                ]
                await self.redis.setex(
                    self._get_session_key(session_id),
                    self.default_ttl,
                    json.dumps(self._serialize_session_data(session_data)),
                )
        except Exception as e:
            logger.error(
                f"Failed to update conversation for session {session_id}: {str(e)}"
            )

    # # Hàm update_conversation (tích hợp với RedisChatMessageHistory)
    # async def update_conversation(self, session_id: str, message: BaseMessage):
    #     """Update conversation history using RedisChatMessageHistory."""
    #     try:
    #         # Lấy settings từ environment variables
    #         redis_host = os.getenv('REDIS_HOST', settings.redis.host)
    #         redis_port = int(os.getenv('REDIS_PORT', getattr(settings.redis, 'port', 6379)))
    #         redis_password = os.getenv('REDIS_PASSWORD')  # Từ Parameter Store

    #         redis_url = f"rediss://:{redis_password}@{redis_host}:{redis_port}/0"
    #         history = RedisChatMessageHistory(
    #             session_id=session_id,
    #             url=redis_url,
    #             ttl=self.default_ttl,
    #         )
    #         if isinstance(message, HumanMessage):
    #             history.add_user_message(message.content)
    #         elif isinstance(message, AIMessage):
    #             history.add_ai_message(message.content)
    #         elif isinstance(message, SystemMessage):
    #             history.add_message(message)
    #         logger.info(f"Updated conversation for session: {session_id}")

    #         # Update session data
    #         session_data = await self.get_session(session_id)
    #         if session_data:
    #             session_data.conversation_history = [
    #                 {"type": msg.__class__.__name__, "content": msg.content}
    #                 for msg in history.messages
    #             ]
    #             await self.redis.setex(
    #                 self._get_session_key(session_id),
    #                 self.default_ttl,
    #                 json.dumps(self._serialize_session_data(session_data)),
    #             )
    #     except Exception as e:
    #         logger.error(f"Failed to update conversation for session {session_id}: {str(e)}")
    #         raise

    async def get_cached_analysis(
        self, session_id: str, query_hash: str, cache_ttl: int
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        if not session_id or not session_id.strip():
            logger.warning("get_cached_analysis called with empty session_id")
            return None

        try:
            cache_key = f"cache:{session_id}:{query_hash}"
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(
                f"Failed to get cached analysis for {session_id}: {str(e)}"
            )
            return None

    async def cache_analysis(
        self, session_id: str, query_hash: str, result: Dict[str, Any]
    ):
        """Cache analysis result."""
        if not session_id or not session_id.strip():
            logger.warning("cache_analysis called with empty session_id")
            return

        try:
            cache_key = f"cache:{session_id}:{query_hash}"
            await self.redis.setex(
                cache_key,
                self.default_ttl,
                json.dumps(result, ensure_ascii=False),
            )
            logger.info(
                f"Cached analysis for session: {session_id}, query: {query_hash}"
            )
        except Exception as e:
            logger.error(
                f"Failed to cache analysis for {session_id}: {str(e)}"
            )

    async def delete_session(self, session_id: str):
        """Delete session with validation."""
        if not session_id or not session_id.strip():
            logger.warning("delete_session called with empty session_id")
            return

        session_id = session_id.strip()

        async with self._lock:
            try:
                session_key = self._get_session_key(session_id)
                history_key = self._get_history_key(session_id)

                result = await self.redis.delete(session_key, history_key)
                if result > 0:
                    logger.info(f"Session and history deleted: {session_id}")
                else:
                    logger.warning(
                        f"Session not found for deletion: {session_id}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to delete session {session_id}: {str(e)}"
                )

    def _get_session_key(self, session_id: str) -> str:
        """Generate session key for Redis."""
        return f"session:{session_id}"

    def _get_history_key(self, session_id: str) -> str:
        """Generate history key for Redis."""
        return f"history:{session_id}"

    def _serialize_session_data(self, session: SessionData) -> Dict[str, Any]:
        """Serialize SessionData to dict for Redis storage."""
        try:
            return {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "analysis_cache": session.analysis_cache or {},
                "metadata": session.metadata or {},
                "conversation_history": session.conversation_history or [],
                "version": "1.2",
            }
        except Exception as e:
            logger.error(
                f"Error serializing session {session.session_id}: {str(e)}"
            )
            return {
                "session_id": session.session_id,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "analysis_cache": {},
                "metadata": {},
                "conversation_history": [],
                "version": "1.2",
            }

    def _deserialize_session_data(self, data: Dict[str, Any]) -> SessionData:
        """Deserialize dict to SessionData with validation."""
        try:
            return SessionData(
                session_id=data.get("session_id", ""),
                created_at=data.get("created_at", time.time()),
                last_accessed=data.get("last_accessed", time.time()),
                analysis_cache=data.get("analysis_cache", {}),
                metadata=data.get("metadata", {}),
                conversation_history=data.get("conversation_history", []),
            )
        except Exception as e:
            logger.error(f"Error deserializing session data: {str(e)}")
            return SessionData(
                session_id=data.get("session_id", "unknown"),
                created_at=time.time(),
                last_accessed=time.time(),
                analysis_cache={},
                metadata={},
                conversation_history=[],
            )

    async def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        try:
            keys = await self.redis.keys("session:*")
            return {
                "active_sessions": len(keys),
                "ttl": self.default_ttl,
                "cleanup_interval": None,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "active_sessions": 0,
                "ttl": self.default_ttl,
                "cleanup_interval": None,
            }
