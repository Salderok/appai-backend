"""Importing this module registers all ORM models with SQLAlchemy's metadata."""

from app.db.models.conversation import Conversation  # noqa: F401
from app.db.models.file import UploadedFile  # noqa: F401
from app.db.models.memory import MemoryItem  # noqa: F401
from app.db.models.message import Message  # noqa: F401
from app.db.models.personality import Personality  # noqa: F401
from app.db.models.user import User  # noqa: F401
