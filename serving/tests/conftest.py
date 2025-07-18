import pytest


@pytest.fixture(scope="session")
def test_image_path():
    return "tests/assets/test_image.jpg"
