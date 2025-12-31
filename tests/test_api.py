import numpy as np
import sys
import os

# Add 'src' to the path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import preprocess_image, app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_homepage():
    """Check if the API is alive"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Banana Ripeness API" in response.json()["message"]

def test_preprocess_shape():
    """
    Check if our preprocessing logic correctly resizes 
    any random image to (1, 299, 299, 3)
    """
    # Create a fake 100x100 random image (in bytes)
    # In a real test we might use a real image file, but for CI speed we mock bytes
    from PIL import Image
    import io
    
    # Generate random pixels
    random_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(random_array)
    
    # Save to bytes buffer (mimicking an UploadFile)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_data = buf.getvalue()
    
    # Run preprocessing
    processed = preprocess_image(byte_data)
    
    # EXPECTATION: Shape should be (1, 299, 299, 3)
    assert processed.shape == (1, 299, 299, 3)
    
    # EXPECTATION: Values should be normalized between 0 and 1
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0