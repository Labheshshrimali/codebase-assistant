import pytest
from pathlib import Path
from app.ingestion.chunker import chunk_file, Chunk

def test_ast_chunking_basic():
    """Test that a simple function is correctly identified as a chunk."""
    code = """def hello():
    print('world')
"""
    # Create a temporary file to test the chunker
    test_file = Path("test_chunk.py")
    test_file.write_text(code)
    
    try:
        chunks = chunk_file(test_file, Path("."))
        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].node_type == "function_definition"
    finally:
        test_file.unlink()

def test_ast_chunking_class_method():
    """Test that methods inside a class are correctly identified."""
    code = """class MyClass:
    def method1(self):
        pass
"""
    test_file = Path("test_class.py")
    test_file.write_text(code)
    
    try:
        chunks = chunk_file(test_file, Path("."))
        # Expect 2 chunks: class header and the method
        assert len(chunks) >= 2
        assert any(c.node_type == "class_header" for c in chunks)
        assert any(c.name == "MyClass.method1" for c in chunks)
    finally:
        test_file.unlink()

def test_chunk_id_generation():
    """Test the Chunk dataclass ID property."""
    c = Chunk("path/to/file.py", 1, 10, "function", "my_func", "code")
    assert c.id == "path/to/file.py:1-10"
    assert c.citation == "path/to/file.py#L1-L10"
