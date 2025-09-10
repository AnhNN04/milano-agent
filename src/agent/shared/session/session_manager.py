import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SessionData:
    session_id: str
    created_at: float
    last_accessed: float
    conversation_history: list
    analysis_cache: Dict[str, Any]
    metadata: Dict[str, Any]


class InMemorySessionManager:
    """In-memory session manager with TTL support"""

    def __init__(self, default_ttl: int = 3600, cleanup_interval: int = 300):
        self.sessions: Dict[str, SessionData] = {}
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.cleanup_task = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start background cleanup task"""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def create_session(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new session with TTL"""
        session_id = str(uuid.uuid4())
        current_time = time.time()

        session_data = SessionData(
            session_id=session_id,
            created_at=current_time,
            last_accessed=current_time,
            conversation_history=[],
            analysis_cache={},
            metadata=metadata or {},
        )

        async with self._lock:
            self.sessions[session_id] = session_data

        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data and update last_accessed"""
        if not session_id:
            return None

        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = time.time()
                return session
            return None

    async def update_conversation(
        self, session_id: str, message: Dict[str, Any]
    ):
        """Add message to conversation history"""
        session = await self.get_session(session_id)
        if session:
            async with self._lock:
                session.conversation_history.append(
                    {**message, "timestamp": time.time()}
                )

    async def cache_analysis(
        self, session_id: str, query_hash: str, result: Dict[str, Any]
    ):
        """Cache analysis result for session"""
        session = await self.get_session(session_id)
        if session:
            async with self._lock:
                session.analysis_cache[query_hash] = {
                    "result": result,
                    "cached_at": time.time(),
                }

    async def get_cached_analysis(
        self, session_id: str, query_hash: str, cache_ttl: int = 300
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis if not expired"""
        session = await self.get_session(session_id)
        if not session:
            return None

        cached = session.analysis_cache.get(query_hash)
        if not cached:
            return None

        if time.time() - cached["cached_at"] > cache_ttl:
            # Cache expired
            async with self._lock:
                session.analysis_cache.pop(query_hash, None)
            return None

        return cached["result"]

    async def delete_session(self, session_id: str):
        """Delete session"""
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []

        async with self._lock:
            for session_id, session_data in self.sessions.items():
                if (
                    current_time - session_data.last_accessed
                    > self.default_ttl
                ):
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self.sessions.pop(session_id, None)

        if expired_sessions:
            print(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_sessions()
        except asyncio.CancelledError:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        current_time = time.time()
        active_sessions = 0
        total_conversations = 0
        total_cached_analyses = 0

        for session in self.sessions.values():
            if current_time - session.last_accessed < self.default_ttl:
                active_sessions += 1
            total_conversations += len(session.conversation_history)
            total_cached_analyses += len(session.analysis_cache)

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_conversations": total_conversations,
            "total_cached_analyses": total_cached_analyses,
        }
