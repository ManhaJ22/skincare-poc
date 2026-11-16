from fastapi.testclient import TestClient

import main


def test_frontend_served_at_app():
    client = TestClient(main.app)
    page = client.get("/app/")
    assert page.status_code == 200
    assert 'src="camera.js"' in page.text
    assert client.get("/app/camera.js").status_code == 200
    # The API routes are not shadowed by the mount.
    assert client.get("/health").json() == {"status": "ok"}


def test_progress_page_served():
    client = TestClient(main.app)
    page = client.get("/app/progress.html")
    assert page.status_code == 200 and 'src="progress.js"' in page.text
    assert client.get("/app/progress.js").status_code == 200
