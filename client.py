import requests
from PIL import Image

# Load image and convert to a format that can be sent over HTTP
image_path = './examples/product-jpeg-500x500.webp'
image = Image.open(image_path)

# Prepare request payload
request_payload = {
    "image": image_path,
    "prompt": "<image>\nPlease describe the image shortly."
}

# Send request to the Ray Serve endpoint
response = requests.post("http://localhost:8000/infer", json=request_payload)

# Print the response from the server
print(response.json())
