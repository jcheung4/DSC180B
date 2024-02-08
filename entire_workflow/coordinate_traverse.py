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
import imageio

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

def get_pic(loc='',size='960x960',heading=0,pitch=5):
    pic_params = {
        'key': API_KEY,
        'location': loc,
        'size': size,
        'heading':heading,
        'pitch':pitch,
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
        temp_img = get_pic(loc=float_to_co(temp),size='20x20').content
        if temp_img != prev_img:
            valid_co += [temp]
            valid_img += [temp_img]
        prev_co = temp
        prev_img = temp_img
    return valid_co

#ang: the angle of the road in radian
#inc: the anticipated heading compared to the road direction
#(0 means it will just look straight down the road)
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
def traverse_collect_images(loc1,loc2,dir='../temp/'):
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
    leftside = [-90]
    rightside = [90]
    
    coors_updated = []

    for index, (value1, value2) in enumerate(coors):
        if index % 2 == 0:  # Skip every other coordinates
            coors_updated.append([value1, value2])
    
    for i,c in enumerate(coors_updated):
        for h in leftside:
            img = get_pic(loc = float_to_co(c),heading = ang_add(ang,h), size = '640x640')
            img_name = f"left{i}_{h}_{c[0]}_{c[1]}.jpg"
            with open(os.path.join(dir, img_name), 'wb') as file:
                file.write(img.content)
        for h in rightside:
            img = get_pic(loc = float_to_co(c),heading = ang_add(ang,h), size = '640x640')
            img_name = f"right{i}_{h}_{c[0]}_{c[1]}.jpg"
            with open(os.path.join(dir, img_name), 'wb') as file:
                file.write(img.content)

def traverse_straight(loc1,loc2,dir='../temp/'):
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
    for i,c in enumerate(coors):
        img = get_pic(loc = float_to_co(c),heading = ang_add(ang,0),size='2560x2560')
        with open(os.path.join(dir, 'test'+str(i)+'.jpg'), 'wb') as file:
            file.write(img.content)

def traverse_curves(locs, dir = '../temp/'):
    idx = 0
    for i in range(len(locs)-1):
        loc1 = locs[i]
        loc2 = locs[i+1]
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
        for _,c in enumerate(coors):
            if _ == 0 and idx>0:
                continue
            img = get_pic(loc = float_to_co(c),heading = ang_add(ang,0),size='960x960')
            with open(os.path.join(dir, 'test'+str(idx).zfill(5)+'.jpg'), 'wb') as file:
                file.write(img.content)
            idx+=1

#dir = '../temp/giftest'
def gif_gen(dir,duration = None):
    images = []
    for f in sorted(os.listdir(dir)):
        images.append(imageio.imread(os.path.join(dir,f)))
    if duration:
        imageio.mimsave(os.path.join(dir,'test.gif'), images, duration = duration)
    else:
        imageio.mimsave(os.path.join(dir,'test.gif'), images)     
#sample code 2
# %%
loc1 = '32.8209644,-117.1861909'
loc2 = '32.8195283,-117.1861259'
#traverse_collect_images(loc1,loc2, './test_2/')

# %%
loc1 = '32.6781445,-117.098631'
loc2 = '32.6787394,-117.0967834'
#traverse_collect_images(loc1,loc2, './test/')

# %%
loc1 = '32.8209644,-117.1861909'
loc2 = '32.8195283,-117.1861259'
#traverse_straight(loc1,loc2)
# %%
#traversing the curvy part and generating gif
locs =[
    '32.8209644,-117.1861909',
    '32.8195283,-117.1861259',
    '32.81854015068618,-117.18570029754645'
]
#traverse_curves(locs)

# %%
#gif_gen('../temp/giftest/fin_gif', duration = 0.08)