from io import BytesIO
from unittest.mock import patch

import web


def test_home_and_health():
    client = web.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Ask what your" in response.data
    assert client.get("/health").json == {"status": "ok"}


@patch("web.answer", return_value=("Retrieved sample passage", []))
def test_question_returns_answer(_answer):
    response = web.app.test_client().post("/ask", data={"question": "What is FAISS?"})

    assert response.status_code == 200
    assert b"Retrieved sample passage" in response.data


def test_text_upload_is_added_to_search():
    client = web.app.test_client()
    response = client.post(
        "/upload",
        data={"document": (BytesIO(b"Upload test document about lunar gardening."), "garden.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Added garden.txt" in response.data