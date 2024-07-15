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
from openai import OpenAI
import tensorflow as tf

import os
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Request 
from fastapi.responses import RedirectResponse
import uvicorn
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from starlette.middleware.sessions import SessionMiddleware
import tensorflow_hub as hub
import re
load_dotenv()

BASE_API_URL = "http://127.0.0.1:7860/api/v1/run"
FLOW_ID = os.getenv("FLOW_ID")
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "ChatInput-EPu3B": {},
  "AstraVectorStoreComponent-NdoG7": {},
  "ParseData-mxIRz": {},
  "Prompt-ts3gs": {},
  "ChatOutput-Nz7mg": {},
  "SplitText-0N7bG": {},
  "File-ouYQ6": {},
  "AstraVectorStoreComponent-arHib": {},
  "OpenAIEmbeddings-n58rH": {},
  "OpenAIEmbeddings-eM4b9": {},
  "OpenAIModel-ZLcsP": {}
}

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
app = FastAPI()
photoix = 0
items = []
placeholder_image = None
CLIENT_SECRETS_FILE = "credentials.json"
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

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

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks=None,
  api_key=None):
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def get_response(msg):
    response = run_flow(
        message=msg,
        endpoint=FLOW_ID,
        output_type="chat",
        input_type="chat",
        tweaks=None,
        api_key=None
    )

    return response["outputs"][0]["outputs"][0]["results"]["message"]["text"]

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

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def load_img(orig_img):
  max_dim = 512
  img_byte_arr = io.BytesIO()
  orig_img.save(img_byte_arr, format='PNG')
  img_byte_arr = img_byte_arr.getvalue()

  out = tf.image.decode_image(img_byte_arr, channels=3)
  img = tf.image.convert_image_dtype(out, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def generate_image(text):
    client = OpenAI(
        api_key=os.getenv('AIML_KEY'),
        base_url="https://api.aimlapi.com/",
        )
    response = client.images.generate(
      model="dall-e-3",
      prompt=text,
      size="1024x1024",
      quality="standard",
      n=1,
    )
    image_url = response.data[0].url
    return image_url

def remove_background(input_img):
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        data={'image_url': input_img,'size': 'auto'},
        headers={'X-Api-Key': os.getenv('BG_KEY')},
    )
    print(response)
    print(response.content)
    return  Image.open(BytesIO(response.content))


def process_image(edited_image, orig_width, orig_height):
    global image1_x, image1_y 
    if edited_image is not None and isinstance(edited_image, dict):
        if 'layers' in edited_image and edited_image['layers']:
            last_layer = edited_image['layers'][-1]
            drawn_pixels = np.where(last_layer[:,:,3] > 0)
            if len(drawn_pixels[0]) > 0:
                min_y, max_y = drawn_pixels[0].min(), drawn_pixels[0].max()
                min_x, max_x = drawn_pixels[1].min(), drawn_pixels[1].max()
                image1_x = min_x
                image1_y = min_y

                image1_x = int(image1_x / 1024 * orig_height)
                image1_y = int(image1_y / 1024 * orig_width)

def combine_images(background_img, overlay_img):
    # Open the background image
    if(isinstance(background_img, np.ndarray)):
        background_img = Image.fromarray(background_img)

    if(isinstance(overlay_img, np.ndarray)):
        overlay_img = Image.fromarray(overlay_img)

    background_img = background_img.convert("RGBA")
    overlay_img = overlay_img.convert("RGBA")

    # Resize overlay to fit on background if needed
    overlay_width = int(background_img.width * 0.3)  # 30% of background width
    overlay_height = int(overlay_img.height * (overlay_width / overlay_img.width))
    overlay_img = overlay_img.resize((overlay_width, overlay_height), Image.LANCZOS)

    # Calculate position to paste overlay (adjust as needed)
    position = (
        (image1_x),
        (image1_y)
    )

    combined = Image.alpha_composite(background_img, Image.new("RGBA", background_img.size))
    combined.paste(overlay_img, position, mask=overlay_img)

    # Save the result
    return combined
def generate_edited_image(orig_img, placement_img, txt):
    image_url = generate_image(txt)
    
    width, height = orig_img.size

    process_image(placement_img, width, height)
    edits = remove_background(image_url)
    return combine_images(orig_img, edits)

def generate_style_image(img, text):
    if(isinstance(img, np.ndarray)):
        img = Image.fromarray(img)
    
    image_url = generate_image(text)
    response = requests.get(image_url)
    style_image = Image.open(BytesIO(response.content))

    content_image = load_img(img)
    style_image = load_img(style_image)

    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)

def interact_chat(message, history):
    prompt = "History: \n"
    for human, ai in history:
        prompt += "Human: " + human + " AI: " + ai + "\n"

    prompt += "Current Message: " + message
    return get_response(prompt)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            button = gr.Button("Login", link="/login")
            user_img = gr.Image(type="pil")

            user_txt = gr.Textbox(label="Enter Text")
            submit_btn = gr.Button(value="Submit")
            chatbot = gr.ChatInterface(interact_chat)
        with gr.Column(scale=1):
            image_editor = gr.ImageEditor(label="Edit Image")
            generated_img = gr.Image(show_label=False)
            edited_img = gr.Image(show_label=False)
            feature_add = gr.Textbox(label="Feature to Add")
            edit_btn = gr.Button(value="Generate Edited Image")

            img_txt = gr.Textbox(label = "Style to Generate:")
            style_btn = gr.Button(value="Generate Styled Image")
            api_resp = gr.Textbox(label = "API Response")
            ad1 = gr.Audio(streaming=True)
            ad2 = gr.Audio(streaming=True)

    edit_btn.click(fn=generate_edited_image, inputs = [user_img, image_editor, feature_add], outputs=edited_img)
    style_btn.click(fn=generate_style_image, inputs = [user_img, img_txt], outputs = generated_img)
    submit_btn.click(
            fn=api_call,
            inputs=[user_img, user_txt],
            outputs=api_resp,
    ).then(
        fn=generate_sound_effect,
        inputs=api_resp,
        outputs=ad2
    ).then(
        fn=generate_and_download_music,
        inputs=api_resp,
        outputs=ad1
    )

app = gr.mount_gradio_app(app, demo, path="/login-demo")

with gr.Blocks() as main_demo:
    with gr.Row():
        with gr.Column(scale=1):
            button = gr.Button("Logout", link="/logout")

            user_img = gr.Image(show_label = False)
            main_demo.load(fn=refresh, inputs=None, outputs=user_img,
                show_progress=False)
            with gr.Row():
                btn_left = gr.Button(value="<", size='sm')
                btn_right = gr.Button(value=">", size='sm')

            user_txt = gr.Textbox(label="Enter Text")
            submit_btn = gr.Button(value="Submit")
            chatbot = gr.ChatInterface(interact_chat)

        with gr.Column(scale=1):
            image_editor = gr.ImageEditor(label="Edit Image")
            generated_img = gr.Image(show_label=False)
            edited_img = gr.Image(show_label=False)

            feature_add = gr.Textbox(label="Feature to Add")
            edit_btn = gr.Button(value="Generate Edited Image")

            img_txt = gr.Textbox(label = "Generate Styled Image")
            style_btn = gr.Button(value="Submit Style prompt")
            api_resp = gr.Textbox(label = "API Response")
            ad1 = gr.Audio(streaming=True)
            ad2 = gr.Audio(streaming=True)
    
    edit_btn.click(fn=generate_edited_image, inputs = [user_img, image_editor, feature_add], outputs=edited_img)

    style_btn.click(fn=generate_style_image, inputs = [user_img, img_txt], outputs = generated_img)
    btn_left.click(fn=mv_right, inputs=None, outputs=user_img)
    btn_right.click(fn=mv_left, inputs=None, outputs=user_img)
    submit_btn.click(
            fn=api_call,
            inputs=[user_img, user_txt],
            outputs=api_resp,
    ).then(
        fn=generate_sound_effect,
        inputs=api_resp,
        outputs=ad2
    ).then(
        fn=generate_and_download_music,
        inputs=api_resp,
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
