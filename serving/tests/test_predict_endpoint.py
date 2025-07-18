import os

from fastapi.testclient import TestClient

from serving.main import app

client = TestClient(app)


def test_predict_endpoint():
    test_image_path = os.path.abspath(
        "serving/tests/assets/cbc9b24f-4ed9-4a69-98bd-c27bdb8bf0d7.jpg"
    )

    response = client.get("/predict/", params={"image_path": test_image_path})

    assert response.status_code == 200
    json_response = response.json()

    assert "detections" in json_response
    for detection in json_response["detections"]:
        assert "label" in detection
        assert "confidence" in detection
        assert "bounding_box" in detection
        assert len(detection["bounding_box"]) == 4
