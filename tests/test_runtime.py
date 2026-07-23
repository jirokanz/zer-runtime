"""Temporary runtime test (fixed import)."""
# This file may be overwritten; we keep a minimal version.
from zeroedge.memory.database import MemoryDB

def test_memory_db_import():
    # Just check that MemoryDB can be imported
    assert MemoryDB is not None
