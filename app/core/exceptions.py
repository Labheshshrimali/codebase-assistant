"""
Custom exceptions for the Codebase Assistant application.
These provide more granular error handling than generic Exceptions.
"""

class CodebaseAssistantError(Exception):
    """Base class for all application-specific errors."""
    pass

class IngestionError(CodebaseAssistantError):
    """Raised when there is a failure during repository cloning or chunking."""
    pass

class IndexingError(CodebaseAssistantError):
    """Raised when there is a failure interacting with the vector database or cache."""
    pass

class RetrievalError(CodebaseAssistantError):
    """Raised when the retrieval pipeline fails to fetch candidates."""
    pass

class GenerationError(CodebaseAssistantError):
    """Raised when the LLM fails to generate an answer."""
    pass
