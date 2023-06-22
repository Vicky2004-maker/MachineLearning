import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# %%
client_id = '931409282eb54f58b85e2ffba245cbdf'
client_secret = 'b932855cf20c4ecd90f1458730a78dae'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#%%
