#%%
import os
import requests
from dotenv import load_dotenv
import json
import base64
import hmac
import urllib.parse as urlparse
import hashlib
import numpy as np

load_dotenv()

API_KEY = os.environ.get("API_KEY")
SECRET = os.environ.get("SECRET")
pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

#%%
def sign_url(input_url=pic_base, secret=SECRET):


    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    url_to_sign = url.path + "?" + url.query
    decoded_key = base64.urlsafe_b64decode(secret)
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
    encoded_signature = base64.urlsafe_b64encode(signature.digest())
    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

def gen_url(params, input_url=pic_base):
    out = input_url
    for i in params:
        out = out+"&"+str(i)+"="+str(params[i])
    return out

def get_pic(loc='',size='960x960',heading =0,pitch=0):
    pic_params = {'key': API_KEY,
        'location': loc,
        'size': size,
        'heading':heading,
        'pitch':5,
    }
    if SECRET:
        pic_response = requests.get(sign_url(input_url=gen_url(pic_params)))
    else:
        pic_response = requests.get(input_url=gen_url(pic_params))
    return pic_response

def float_to_co(arr):
    return str(arr[0])+','+str(arr[1])

#generate list of coordinates 
#return list filled with list length of 2
def unique_coordinates(start,ang,length,interval = 0.00005):
    prev_co = start
    prev_img = get_pic(loc=float_to_co(prev_co)).content
    # list to store coordinates that generate unique images
    valid_co = [prev_co]
    valid_img = [prev_img]
    for _ in range(int(length/interval)):
        temp = [prev_co[0] + interval*np.sin(ang), prev_co[1] + interval*np.cos(ang)]
        temp_img = get_pic(loc=float_to_co(temp),size='100x100').content
        if temp_img != prev_img:
            valid_co += [temp]
            valid_img += [temp_img]
        prev_co = temp
        prev_img = temp_img
    return valid_co

#ang: the angle of the road in radian
#inc: the anticipated heading compared to the road direction
def ang_add(ang, inc):
    ang_deg = (ang*180/np.pi)
    if ang_deg > 90:
        ang_deg = 450 - ang_deg
    else:
        ang_deg = 90 - ang_deg
    out = ang_deg +inc
    if out >360:
        out -= 360
    elif out <0:
        out+360
    return out

#loc1 and loc2 should be string
def traverse_collect_images(loc1,loc2):
    one = loc1.split(',')
    one[0] = float(one[0])
    one[1] = float(one[1])
    two = loc2.split(',')
    two[0] = float(two[0])
    two[1] = float(two[1])
    #first number indicates y coordinate and second number indicate x coordinate
    ygap = two[0] - one[0]
    xgap = two[1] - one[1]
    length = (xgap**2+ygap**2)**0.5
    ang = np.arctan2(ygap,xgap)
    coors = unique_coordinates(one,ang,length)
    leftside = [-150,-90,-30]
    rightside = [150,90,30]
    for i,c in enumerate(coors):
        for h in leftside:
            img = get_pic(loc = float_to_co(c),heading = ang_add(ang,h))
            with open('../temp/test'+str(i)+'_'+str(h)+'.jpg', 'wb') as file:
                file.write(img.content)
        for h in rightside:
            img = get_pic(loc = float_to_co(c),heading = ang_add(ang,h))
            with open('../temp/test'+str(i)+'_'+str(h)+'.jpg', 'wb') as file:
                file.write(img.content)

#sample code, provided loc1, loc2. This just to show the code work. 
#The heading is set to 0 which to see if the heading is following the road.
"""
loc1='32.85179322509682,-117.19691677340164'
loc2='32.8514507286907,-117.19493193834228'
one = loc1.split(',')
one[0] = float(one[0])
one[1] = float(one[1])
two = loc2.split(',')
two[0] = float(two[0])
two[1] = float(two[1])
#first number indicates y coordinate and second number indicate x coordinate
ygap = two[0] - one[0]
xgap = two[1] - one[1]
length = (xgap**2+ygap**2)**0.5
ang = np.arctan2(ygap,xgap)
coors = unique_coordinates(one,ang,length)
leftside = [-150,-90,-30]
rightside = [150,90,30]
for i,c in enumerate(coors):
    img = get_pic(loc = float_to_co(c),heading = ang_add(ang,0))
    with open('../temp/test'+str(i)+'.jpg', 'wb') as file:
        file.write(img.content)
"""
#sample code 2
"""
loc1='32.85179322509682,-117.19691677340164'
loc2='32.8514507286907,-117.19493193834228'
traverse_collect_images(loc1,loc2)
"""
# %%
