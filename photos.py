import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from flask import Flask, redirect, url_for, session, request, render_template_string
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
CLIENT_SECRETS_FILE = "credentials.json"

SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
API_SERVICE_NAME = 'photoslibrary'
API_VERSION = 'v1'

# Set up the OAuth 2.0 flow
flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    redirect_uri='http://localhost:8080/auth/google/callback'
)

@app.route('/')
def index():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))

    credentials = Credentials(**session['credentials'])
    service = build(API_SERVICE_NAME, API_VERSION, credentials=credentials, static_discovery=False)

    # Access the user's photos
    results = service.mediaItems().list(pageSize=10).execute()
    items = results.get('mediaItems', [])

    if not items:
        photos = 'No photos found.'
    else:
        photos = '<ul>'
        for item in items:
            photos += f'<li><img src="{item["baseUrl"]}=w200-h200" alt="{item["filename"]}"></li>'
        photos += '</ul>'

    return render_template_string("""
        <h1>Your Google Photos</h1>
        {{ photos|safe }}
    """, photos=photos)

@app.route('/authorize')
def authorize():
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)

@app.route('/auth/google/callback')
def oauth2callback():
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    return redirect(url_for('index'))

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

if __name__ == '__main__':
    app.run(port=8080, debug=True)
