import os
import requests
from dotenv import load_dotenv
import hashlib
import hmac
import base64
import urllib.parse as urlparse

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("API_KEY")
digital_signature = os.getenv("SECRET")

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
    Args:
        input_url - The URL to sign
        secret    - Your URL signing secret
    Returns:
        The signed request URL
    """
    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")
    
    # Parse the input URL
    url = urlparse.urlparse(input_url)
    
    # Concatenate the URL path and query to sign
    url_to_sign = url.path + "?" + url.query
    
    # Decode the secret key
    decoded_key = base64.urlsafe_b64decode(secret)
    
    # Create the signature
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
    
    # Encode the signature in base64
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    
    # Reconstruct the original URL with scheme and netloc
    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Append the signature to the URL as a query parameter
    return original_url + "&signature=" + encoded_signature.decode()

def fetch_image(lat, lon, heading, fov, api_key=api_key, secret_key=digital_signature):
    """Fetches an image from Google Street View API"""
    # Google Street View API endpoint
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    # Parameters for the API request
    params = {
        'size': '600x400',
        'location': f'{lat},{lon}',
        'heading': str(heading),
        'fov': str(fov),
        'key': api_key
    }
    
    # Construct the URL with parameters
    url = base_url + "?"
    for key, value in params.items():
        url += f"{key}={value}&"
    url = url.rstrip("&")
    
    # Sign the URL with your secret key
    signed_url = sign_url(url, secret_key)
    
    # Send the request to the API and return the image content
    response = requests.get(signed_url)
    return response.content

def save_image(image_content, filename):
    """Saves the image content to a file"""
    # Directory to save images
    directory = '../../data/images/'
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    # Full path for the image file
    file_path = os.path.join(directory, filename)
    
    # Write the image content to the file
    with open(file_path, 'wb') as file:
        file.write(image_content)
        
    # Log the saved image path
    print(f"Image saved at: {file_path}")

def capture_and_save_pole_images(pole_type, coordinates, zoom_levels):
    """Captures and saves images of poles of a given type"""
    for lat, lon, starting_angle in coordinates:
        angles = [(starting_angle + 15 * i) % 360 for i in range(5)]
        for angle in angles:
            for zoom in zoom_levels:
                image_content = fetch_image(lat, lon, angle, zoom, api_key, digital_signature)
                filename = f"derek_{pole_type}_lat{lat}_lon{lon}_angle{angle}_zoom{zoom}.jpg"
                save_image(image_content, filename)
                print(f"Saved: {filename}")

def derek_create_images():
    # Define zoom levels for image capture
    zoom_levels = [175, 150, 125, 100, 75]
    
    # Metal pole coordinates
    metal_pole_coordinates = [
        # Each tuple contains latitude, longitude, and starting angle
        (32.9512618, -117.2474648, 70), 
        (32.9587855, -117.2665225, 265), 
        (32.7506093, -117.2162704, 50),
        (32.8029464, -116.9976139, 55), 
        (32.7951143, -116.9971548, 75)
    ]

    # Wooden pole coordinates
    wooden_pole_coordinates = [
        # Each tuple contains latitude, longitude, and starting angle
        (32.9493593, -117.2474197, 90), 
        (32.9498758, -117.2474622, 340), 
        (32.9520803, -117.2474362, 350), 
        (32.953285, -117.2473199, 150), 
        (32.9530089, -117.2465422, 60)
    ]

    # Capture and save images of metal and wooden poles
    capture_and_save_pole_images("metal", metal_pole_coordinates, zoom_levels)
    capture_and_save_pole_images("wood", wooden_pole_coordinates, zoom_levels)

if __name__ == "__main__":
    derek_create_images()