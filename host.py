import requests
import base64
from elevenlabs.client import ElevenLabs
import gradio as gr
import requests
import json
import base64
from PIL import Image
import io
from io import BytesIO
import numpy as np

import os
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Request 
from fastapi.responses import RedirectResponse
import uvicorn
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from starlette.middleware.sessions import SessionMiddleware

import re
load_dotenv()

app = FastAPI()
photoix = 0
items = []
placeholder_image = None
CLIENT_SECRETS_FILE = "credentials.json"

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
API_SERVICE_NAME = 'photoslibrary'
API_VERSION = 'v1'

# Set up the OAuth 2.0 flow
flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    redirect_uri='http://127.0.0.1:8000/auth'
)

google_credentials = None

elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
effects_path = "effects.mp3"
song_path = "audio_file.mp3"

SECRET_KEY = os.environ.get('SECRET_KEY') or "a_very_secret_key"
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
# Assuming you have an image object called 'img'
def get_user(request: Request):
    user = request.session.get('credentials')
    if user and google_credentials:
        return user
    return None

@app.get('/')
def public(user: dict = Depends(get_user)):
    global items
    if user:
        get_google_photo()
        print(len(items))
        # Access the user's photos
        return RedirectResponse(url='/gradio')
    else:
        return RedirectResponse(url='/login-demo')

@app.route('/logout')
async def logout(request: Request):
    request.session.pop('credentials', None)
    return RedirectResponse(url='/')

@app.route('/login')
async def login(request: Request):
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    request.session['state'] = state
    return RedirectResponse(authorization_url)


@app.route('/auth')
async def auth(request: Request):
    global google_credentials
    flow.fetch_token(authorization_response=str(request.url._url))

    credentials = flow.credentials
    request.session['credentials'] = credentials_to_dict(credentials)
    google_credentials = request.session['credentials']
    return RedirectResponse(url='/')

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

system_prompt = "Based on the scenery in this image, generate a description of the sounds that you think could be heard if you were in the image. For example, if there is a river or stream in the background, please provide a detailed description of the sound of water flowing down. The description of the sounds in the image should make the user feel like they are a part of the image, and reliving the moment. If there is a user prompt, ensure to include those instructions as well. Make sure to limit your response to 20 words."


def generate_sound_effect(text: str):

    result = elevenlabs.text_to_sound_effects.convert(
        text=text,
        duration_seconds=10,  
        prompt_influence=0.3,  
    )

    for chunk in result:
        yield chunk

def generate_and_download_music(prompt):

    import requests

    url = "https://api.aimlapi.com/generate"
    headers = {
        "Authorization": f"Bearer {os.getenv('AIML_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "make_instrumental": True,
        "wait_audio": True
    }
    response = requests.post(url, json=payload, headers=headers)
    print(response)
    response_json = response.json()
    print(response_json)
    song_id1 = response_json[0]['id']
    url1 = response_json[0]['audio_url']

    song_id2 = response_json[1]['id']
    url2 = response_json[1]['audio_url']

    # Print the important data from the first response
    print(f"ID (1): {song_id1}")
    print(f"Streaming URL - Song 1: {url1}")

    print(f"ID (2): {song_id2}")
    print(f"Streaming URL - Song 2: {url2}")

    # Stream the audio content
    audio_response = requests.get(url2, stream=True)

    for chunk in audio_response.iter_content(chunk_size=4096):
        if chunk:
            yield chunk

def api_call(img, txt):
    if(isinstance(img, np.ndarray)):
        img = Image.fromarray(img)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    url = "https://api.imgbb.com/1/upload"
    params = {
        'expiration': '600',
        'key': f'{os.getenv("IMGUR_KEY")}'
    }
    files = {
        "image": ("image.jpg", img_byte_arr, "image/jpeg")
    }
    

    img_upload_response = requests.post(url, params=params, files=files)
    img_upload_response = img_upload_response.json()
    image_path = img_upload_response['data']['url']

    url = "https://api.aimlapi.com/chat/completions"
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {os.getenv('AIML_KEY')}"

    }
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
              {"type": "text", "text": f"{system_prompt} : User prompt: {txt}"},
            {"type": "image_url", "image_url": {"url": f"{image_path}"}}
          ]
        }
      ],
      "max_tokens": 300
    }

    response = requests.post(url, headers=headers, json=payload)

    response_json = response.json()

    content = response_json['choices'][0]['message']['content']
    return content 

def get_google_photo():
    global items, photoix
    credentials = Credentials(**google_credentials)
    service = build(API_SERVICE_NAME, API_VERSION, credentials=credentials, static_discovery=False)

    # Access the user's photos
    results = service.mediaItems().list(pageSize=10).execute()
    items = results.get('mediaItems', [])
    print(len(items))

def refresh():
    global items, photoix
    response = requests.get(items[photoix]['baseUrl'])
    image_data = response.content
    image = Image.open(BytesIO(image_data))
    return image

def mv_right():
    global photoix
    photoix = (photoix + 1) % len(items)
    return refresh()

def mv_left():
    global photoix
    photoix = (photoix - 1) % len(items)
    return refresh()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            button = gr.Button("Login", link="/login")
            user_img = gr.Image(type="pil")

            user_txt = gr.Textbox(label="Enter Text")
            submit_btn = gr.Button(value="Submit")
        with gr.Column(scale=1):
            txt2 = gr.Textbox(label = "API Response")
            ad1 = gr.Audio(streaming=True)
            ad2 = gr.Audio(streaming=True)
   
    submit_btn.click(
            fn=api_call,
            inputs=[user_img, user_txt],
            outputs=txt2
    ).then(
        fn=generate_sound_effect,
        inputs=txt2,
        outputs=ad2
    ).then(
        fn=generate_and_download_music,
        inputs=txt2,
        outputs=ad1
    )

app = gr.mount_gradio_app(app, demo, path="/login-demo")

with gr.Blocks() as main_demo:
    with gr.Row():
        with gr.Column(scale=1):
            user_img = gr.Image(show_label = False)
            main_demo.load(fn=refresh, inputs=None, outputs=user_img,
                show_progress=False)
            with gr.Row():
                btn_left = gr.Button(value="<", size='sm')
                btn_right = gr.Button(value=">", size='sm')

            user_txt = gr.Textbox(label="Enter Text")
            submit_btn = gr.Button(value="Submit")
        with gr.Column(scale=1):
            txt2 = gr.Textbox(label = "API Response")
            ad1 = gr.Audio(streaming=True)
            ad2 = gr.Audio(streaming=True)
    
    btn_left.click(fn=mv_right, inputs=None, outputs=user_img)
    btn_right.click(fn=mv_left, inputs=None, outputs=user_img)
    submit_btn.click(
            fn=api_call,
            inputs=[user_img, user_txt],
            outputs=txt2
    ).then(
        fn=generate_sound_effect,
        inputs=txt2,
        outputs=ad2
    ).then(
        fn=generate_and_download_music,
        inputs=txt2,
        outputs=ad1
    )

app = gr.mount_gradio_app(app, main_demo, path="/gradio", auth_dependency=get_user)
# demo = gr.Interface(
    # fn=api_call,
    # inputs=[gr.Image(type="pil"), gr.Textbox(label="Enter Text")],
    # outputs=[gr.Textbox(label = "API Response"), gr.Audio(type="filepath"), gr.Audio(type="filepath")]
# )

if __name__ == '__main__':
    uvicorn.run(app)
