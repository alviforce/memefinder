"""
Shared fixtures for MemFinder tests.
"""
import sys
import pytest
import threading
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """
    Isolated SQLite database for each test.
    Patches DB_PATH and resets thread-local connections.
    """
    import database
    monkeypatch.setattr("database.DB_PATH", tmp_path / "test.db")
    # Reset thread-local connection so the new path is used
    database._local = threading.local()
    database.init_db()
    yield database
    # Cleanup: close connection if open
    if hasattr(database._local, "conn") and database._local.conn:
        database._local.conn.close()
        database._local.conn = None


@pytest.fixture
def sample_thumbnail():
    """A tiny valid base64 JPEG thumbnail string."""
    import base64
    import io
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


@pytest.fixture
def sample_image():
    """A small PIL image for CLIP tests."""
    from PIL import Image
    return Image.new("RGB", (64, 64), color=(100, 150, 200))
