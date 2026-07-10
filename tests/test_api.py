import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Verify the health check returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_ingest_invalid_url():
    """Verify that non-github URLs are rejected."""
    response = client.post("/ingest", json={"repo_url": "https://google.com"})
    assert response.status_code == 400
    assert "Only https://github.com/ URLs are supported" in response.json()["detail"]

def test_ask_without_index():
    """Verify that asking about an unindexed repo returns a helpful error."""
    response = client.post("/ask", json={"repo": "non_existent_repo", "question": "hello"})
    assert response.status_code == 200 # The current API returns 200 with error msg
    assert "error" in response.json()
