# %%
from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO
import IPython.display as display
import json

load_dotenv()
# %%
import hashlib
import hmac
import base64
import urllib.parse as urlparse


def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
    """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

# %%
def get_image(lat, long, size = '640x400', fov = 90, heading = 0, pitch = 0, show = False, save_path = ''):
    api_key = os.getenv('API_KEY')
    secret = os.getenv('SECRET')

    # Create the Street View URL
    street_view_url = f"https://maps.googleapis.com/maps/api/streetview?size={size}&location={lat},{long}&fov={fov}&heading={heading}&pitch={pitch}&key={api_key}"
    signed_url = sign_url(street_view_url, secret)

    # Make the HTTP request to fetch the Street View image
    response = requests.get(signed_url)

    if response.status_code == 200:
        if show:
            img = Image.open(BytesIO(response.content))
            display.display(img)
        if save_path:
            with open(save_path, 'wb') as image_file:
                image_file.write(response.content)
    else:
        print(f"Error: {response.status_code} - Failed to fetch Street View image.")

"""
# %%
with open('kevin_structures.json', 'r') as file:
    all_structures = json.load(file)
    
parent_dir = 'images'

if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
    
for name, struc in all_structures.items():
    #subfolder_name = os.path.join(parent_dir, name)
    #if not os.path.exists(subfolder_name):
    #    os.mkdir(subfolder_name)
    print(name)
    for i in range(len(struc['lat'])):
        for index, zoom in enumerate([5, 10, 15]):
            get_image(
                lat = struc['lat'][i],
                long = struc['long'][i],
                fov = struc['fov'] + zoom,
                heading = struc['heading'][i],
                pitch = struc['pitch'],
                #show = True,
                save_path = os.path.join(parent_dir, f'kevin_{name}_{i}_{index}.jpg')
            )
# %%
"""

def kevin_create_images(data_file='', output_path=''):
    with open(data_file, 'r') as file:
        all_structures = json.load(file)
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            
        for name, struc in all_structures.items():
            for i in range(len(struc['lat'])):
                for index, zoom in enumerate([5, 10, 15]):
                    get_image(
                        lat=struc['lat'][i],
                        long=struc['long'][i],
                        fov=struc['fov'] + zoom,
                        heading=struc['heading'][i],
                        pitch=struc['pitch'],
                        show=False,
                        save_path=os.path.join(output_path, f'kevin_images_{name}_{i}_{index}.jpg')
                    )

if __name__ == "__main__":
    kevin_create_images(data_file='../../data/kevin_structures.json', output_path='../../data/images/')