import tempfile
import unittest
from pathlib import Path

from src.gui.database import ChatRepository


class ChatRepositoryTests(unittest.TestCase):
    def test_update_session_title(self) -> None:
        db_path = Path(tempfile.mkdtemp()) / "chat.db"
        repository = ChatRepository(str(db_path))
        session_id = repository.create_session()

        repository.update_session_title(session_id, "Helpful title")

        self.assertEqual(repository.get_session_title(session_id), "Helpful title")
        self.assertEqual(repository.list_sessions()[0]["title"], "Helpful title")


if __name__ == "__main__":
    unittest.main()
