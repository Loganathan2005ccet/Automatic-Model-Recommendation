import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline
from textblob import TextBlob
import plotly.graph_objects as go
import random

# Spotify API Credentials (replace with your own credentials)
SPOTIPY_CLIENT_ID = '91e7e71cd8884100a4aab594b72cd21f'
SPOTIPY_CLIENT_SECRET = '2809a68c966e4ac1a8d68f90a0e7a929'
SPOTIPY_REDIRECT_URI = 'http://localhost:8501/'

# Authenticate with Spotify
auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# More nuanced mood detection using a larger set of moods
mood_genre_map = {
    'happy': ['pop', 'dance', 'party'],
    'sad': ['blues', 'classical', 'soft rock'],
    'excited': ['rock', 'edm', 'hip hop'],
    'relaxed': ['jazz', 'acoustic', 'chill'],
    'angry': ['metal', 'punk', 'rock'],
    'romantic': ['soul', 'R&B', 'soft rock'],
    'nostalgic': ['oldies', 'classical', 'country'],
}
# Initialize the sentiment analysis model (you can use 'distilbert-base-uncased' or 'roberta-base')
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased")


# Define your mood features mapping
mood_features_map = {
    'happy': {'valence': 0.9, 'energy': 0.7, 'danceability': 0.6},
    'sad': {'valence': 0.2, 'energy': 0.1, 'acousticness': 0.6},
    'excited': {'energy': 0.95, 'danceability': 0.9, 'tempo': 130},
    'relaxed': {'energy': 0.3, 'acousticness': 0.9, 'valence': 0.5},
    'angry': {'energy': 0.85, 'tempo': 130, 'valence': 0.4},
    'romantic': {'valence': 0.8, 'energy': 0.4, 'danceability': 0.5},
    'nostalgic': {'valence': 0.65, 'acousticness': 0.85, 'energy': 0.4},
}

# Initialize sentiment analysis model
analyzer_basic = TextBlob
sentiment_analysis_pipeline = pipeline("sentiment-analysis")


# Enhanced mood analysis function
def enhanced_analyze_feelings(user_feelings):
    analysis_advanced = sentiment_analysis_pipeline(user_feelings)
    mood_detected = 'neutral'

    # Determine mood based on sentiment analysis
    if analysis_advanced[0]['label'] == 'POSITIVE':
        mood_detected = 'happy'
    elif analysis_advanced[0]['label'] == 'NEGATIVE':
        mood_detected = 'sad'
    else:
        mood_detected = 'relaxed'
    # Additional contextual keywords can be checked here
    if 'exciting' in user_feelings.lower() or 'thrilled' in user_feelings.lower():
        mood_detected = 'excited'
    elif 'relax' in user_feelings.lower() or 'calm' in user_feelings.lower():
        mood_detected = 'relaxed'
    elif 'angry' in user_feelings.lower() or 'furious' in user_feelings.lower():
        mood_detected = 'angry'
    elif 'love' in user_feelings.lower() or 'romantic' in user_feelings.lower():
        mood_detected = 'romantic'
    elif 'remember' in user_feelings.lower() or 'nostalgia' in user_feelings.lower():
        mood_detected = 'nostalgic'

    print("Detected Mood:", mood_detected)
    return mood_detected

# Function to create a speedometer visualization
def create_speedometer(mood):
    mood_scores = {
        'happy': 80,
        'excited': 100,
        'relaxed': 70,
        'sad': 30,
        'angry': 20,
        'romantic': 60,
        'nostalgic': 50,
        'neutral': 40,
    }
    
    score = mood_scores.get(mood, 0)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Mood Intensity"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "cyan"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))

    return fig

# Function to get updated age group
def get_age_group(age):
    if age < 13:
        return 'child'
    elif age < 18:
        return 'teen'
    elif age < 30:
        return 'young adult'
    elif age < 45:
        return 'adult'
    elif age < 60:
        return 'middle-aged'
    else:
        return 'senior'

# Updated function to get age-specific genres
age_genre_map = {
    'child': ['children', 'animated'],
    'teen': ['pop', 'hip hop', 'k-pop'],
    'young adult': ['rock', 'edm', 'indie'],
    'adult': ['jazz', 'blues', 'soul'],
    'middle-aged': ['classic rock', 'soft rock', 'country'],
    'senior': ['oldies', 'classical', 'nostalgic']
}


# Language to artist mapping
language_artists_map = {
    'English': [
        'https://open.spotify.com/artist/3TVXtAsR1Inumwj472S9r4',  # Drake
        'https://open.spotify.com/artist/1uNFoZAHBGtllmzznpCI3s',  # Justin Bieber
        'https://open.spotify.com/artist/6eUKZXaKkcviH0Ku9w2n3V',  # Ed Sheeran
        'https://open.spotify.com/artist/53XhwfbYqKCa1cC15pYq2q',  # Imagine Dragons
        'https://open.spotify.com/artist/66CXWjxzNUsdJxJ2JdwvnR',  # Ariana Grande
    ],
    'Hindi': [
        'https://open.spotify.com/artist/5f4QpKfy7ptCHwTqspnSJI',   # Arijit Singh
        'https://open.spotify.com/artist/1wRPtKGflJrBx9BmLsSwlU',  # Neha Kakkar
        'https://open.spotify.com/artist/3TLpT1Jz5kKBLkxtnInC9z',  # Badshah
        'https://open.spotify.com/artist/4o8RzJ9Ksj5mj25byyUlKz',  # Shreya Ghoshal
        'https://open.spotify.com/artist/4YXycRbyyAE0wozTk7QMEq',  # Jubin Nautiyal
    ],
    'Tamil': [
        'https://open.spotify.com/artist/6AiX12wXdXFoGJ2vk8zBjy',  # AR Rahman
        'https://open.spotify.com/artist/1VX5VW0BEaWYlJwA7k6J9n',  # D. Imman
        'https://open.spotify.com/artist/4udfTWwtZsXa9peEt8qrN1',  # Anirudh Ravichander
        'https://open.spotify.com/artist/3PeHfG9GK1eDFwlFnyIEYk',  # Sid Sriram
        'https://open.spotify.com/artist/7ceIhLzBzjuY42cBLQl3oa',  # Karthik
    ],
    'Telugu': [
        'https://open.spotify.com/artist/7qjJw7ZM2ekDSahLXPjIlN?si=h8nzxQMiTCOqiZE-0Mupqw',  # Sid Sriram
        'https://open.spotify.com/artist/5sSzCxHtgL82pYDvx2QyEU?si=fZ_sRPa1SNeI212PY3UNOA',  # Devi Sri Prasad (DSP)
        'https://open.spotify.com/artist/4IKVDbCSBTxBeAsMKjAuTs?si=ccf6b114e9674880',  # Armaan Malik
        'https://open.spotify.com/artist/2ae6PxICSOZHvjqiCcgon8?si=36de93f9debd4811',  # S. P. Balasubrahmanyam
        'https://open.spotify.com/artist/12l1SqSNsg2mI2IcXpPWjR?si=2412d7524f58415f',  #keeravani
    ],
    'Kannada': [
        'https://open.spotify.com/artist/4iA6bUhiZyvRKJf4FNVX39?si=AIoAvzBiSVaj6yU_GP9dwQ',  # Vijay Prakash
        'https://open.spotify.com/artist/3IX32wm6CoEIYovZ0VcjBJ?si=746ed5f99ac94d3c',  # Arjun Janya
        'https://open.spotify.com/artist/0ZnBmsYz6ImvXdfUglJEWA?si=5a2ef79e796a4e39',  # Rajesh Krishnan
        'https://open.spotify.com/artist/0oOet2f43PA68X5RxKobEy?si=a2153744982a4554',  # Shreya Ghoshal (Kannada songs)
        'https://open.spotify.com/artist/002yVW3Yn595KWy74buQ1k?si=d57b0d3194a341a2',  # Chandan Shetty
    ],
    'Malayalam': [
        'https://open.spotify.com/artist/4JXqxFqi9dxlsiXKZhKvzB?si=b4ae2289fa194bc3',  # KS Harisankar
        'https://open.spotify.com/artist/2wPsNCwhEGb0KvChZ5DD52?si=4926f51a7d4b4925',  # jesuthas
        'https://open.spotify.com/artist/2oJbFGuxu5d9xvez6yHvFh?si=54c4654f22b544fe',  # Sithara Krishnakumar
        'https://open.spotify.com/artist/2NoJ7NuNs9nyj8Thoh1kbu?si=e393d74bbc094b23',  # MG Sreekumar
        'https://open.spotify.com/artist/4xlqU0G9EloUPHL1qlmWY6?si=025cd9ed52814fe2',  # Gopi Sundar
    ],
    'Bengali': [
        'https://open.spotify.com/artist/4YRxDV8wJFPHPTeXepOstw?si=f1601366e7fa40d4',  # Arijit Singh (Bengali songs)
        'https://open.spotify.com/artist/2kkQthS9OLpK4UqNWYqoVl?si=1712ff4b567644dc',  # Shreya Ghoshal (Bengali songs)
        'https://open.spotify.com/artist/5LZ894xYE9MG1sal0gjt5L?si=0a11bfb391e1425e',  # Rupam Islam
        'https://open.spotify.com/artist/5LZ894xYE9MG1sal0gjt5L?si=3ea35e7227dd4ca9',  # Nachiketa Chakraborty
        'https://open.spotify.com/artist/7gjiYwM6O5sNuYBaCdpCXA?si=d06a9d8d4c30476f',  # Iman Chakraborty
    ],
    'Punjabi': [
        'https://open.spotify.com/artist/4PULA4EFzYTrxYvOVlwpiQ?si=d18f8b4496644c4a',  # Sidhu Moose Wala
        'https://open.spotify.com/artist/6LEG9Ld1aLImEFEVHdWNSB?si=9d27c28687e541d0',  # AP Dhillon
        'https://open.spotify.com/artist/2FKWNmZWDBZR4dE5KX4plR?si=fffc8f5cae1643c9',  # Diljit Dosanjh
        'https://open.spotify.com/artist/2RlWC7XKizSOsZ8F3uGi59?si=fdc1693153b348e9',  # Ammy Virk
        'https://open.spotify.com/artist/56SjZARoEvag3RoKWIb16j?si=7e523694416d45b7',  # B Praak
    ],
    'Marathi': [
        'https://open.spotify.com/artist/5fvTHKKzW44A9867nPDocM?si=dc0124f70ea74b63',  # Ajay-Atul
        'https://open.spotify.com/artist/6mxY3ekITToaEK2XGtaock?si=ee11126ea0dd4fa3',  # Avadhoot Gupte
        'https://open.spotify.com/artist/1SJOL9HJ08YOn92lFcYf8a?si=61f54fa0ba1342d5',  # Shankar Mahadevan (Marathi songs)
        'https://open.spotify.com/artist/4B9efXsA6sv4w3vts8E0T7?si=21fd585a21fa4f67',  # Sandeep Khare
        'https://open.spotify.com/artist/2zGP2SUtwsDhdyYzf0kKp8?si=060d9e7308ec4bb3',  # Vaishali Samant
    ],
    'Gujarati': [
        'https://open.spotify.com/artist/1SJOL9HJ08YOn92lFcYf8a?si=61f54fa0ba1342d5',  # Kirtidan Gadhvi
        'https://open.spotify.com/artist/7MAlFea251zaprQFjwvYaL?si=164ea31fb4c54944',  # Aishwarya Majmudar
        'https://open.spotify.com/artist/26qILArN7gTOjFRTbOTKbJ?si=c6f3ce5b41b74651',  # Kinjal Dave
        'https://open.spotify.com/artist/2Hms1YhTKaXQ5yZPGKztXe',  # Jigardan Gadhavi
        'https://open.spotify.com/artist/6SLyfZLFgTmvDJh8XhzESU?si=2c5273a79d694ff6',  # Hemant Chauhan
    ],
    'French': [
        'https://open.spotify.com/artist/1CoZyIxLU3rDq3SnE8wVtw',  # Stromae
        'https://open.spotify.com/artist/2SOB6LNBtdcFfwDkSMthF6',  # Christine and the Queens
        'https://open.spotify.com/artist/7o46KLY6Fq8nT4Gx4G7M3B',  # Angèle
        'https://open.spotify.com/artist/2k74oP4c4bP0o4PtEDU8bT',  # Amel Bent
        'https://open.spotify.com/artist/4kT1TpDFl0zmEtAn2kVDAH',  # Louane
    ],
    'Spanish': [
        'https://open.spotify.com/artist/1mcTU81TzQhprhouKaTkpq',  # Bad Bunny
        'https://open.spotify.com/artist/4tZ0Vf1azXrIhK1RlGznpB',  # Shakira
        'https://open.spotify.com/artist/6G2I01zGXLUAMzWzOENkpK',  # J Balvin
        'https://open.spotify.com/artist/2ZZFj75Ip4Emi1cK0tM7KA',  # Rosalía
        'https://open.spotify.com/artist/6l8qDku5BbYYBWkMf1jQz1',  # Enrique Iglesias
    ],
    'German': [
        'https://open.spotify.com/artist/5K4W6rqBFWDnAN6FQUkS6x',  # Kanye West (for German market)
        'https://open.spotify.com/artist/2IZgYxi5Vm8hvTHf7nJY3D',  # Tim Bendzko
        'https://open.spotify.com/artist/5t4V0LFqojL5DRh0ST7tUk',  # Helene Fischer
        'https://open.spotify.com/artist/1UuM4u7Oq0RXiHwc6b0zZF',  # Mark Forster
        'https://open.spotify.com/artist/0w9cmYbwMw7hNOFLK0g5mv',  # Peter Fox
    ],
    'Brazilian Portuguese': [
        'https://open.spotify.com/artist/7tYKF4w9nC0nq9CsPZTHyP',  # Anitta
        'https://open.spotify.com/artist/1Nq2GRoaB5AqGEjSBcYJW3',  # Jorge & Mateus
        'https://open.spotify.com/artist/3yUOLpRnsuW3RCbT3pTnbW',  # Alok
        'https://open.spotify.com/artist/6UjHAlHg3D5Gh2UBhbANHo',  # Zeca Pagodinho
        'https://open.spotify.com/artist/5g8wE2IN7cYjZgB9IfA7U6',  # Ivete Sangalo
    ],
    'Japanese': [
        'https://open.spotify.com/artist/1snhtMLeb2DYoMOcVbb8iB',  # Kenshi Yonezu
        'https://open.spotify.com/artist/6T3F5Nf3Q3eYRV3KnqO6jC',  # Utada Hikaru
        'https://open.spotify.com/artist/5b9k8uW4hN1KpQ28qqf7xH',  # Arashi
        'https://open.spotify.com/artist/2auG9R2ZGO3I4WKh36P6Rp',  # LiSA
        'https://open.spotify.com/artist/1lX0rxhKZBSWuEpZOWxwjh',  # Namie Amuro
    ],
}


# Enhanced recommendations function
def get_enhanced_recommendations(mood, language, region, age):
    mood_features = mood_features_map.get(mood)
    genre = random.choice(mood_genre_map[mood])
    
    age_group = get_age_group(age)
    age_genres = age_genre_map.get(age_group, [])

    seed_artists = language_artists_map.get(language, None)
    recommendations = None

    # First attempt to get recommendations using seed artists
    if seed_artists:
        try:
            recommendations = sp.recommendations(seed_artists=seed_artists, limit=50, market=region, **mood_features)
        except Exception as e:
            st.write(f"Error fetching recommendations with seed artists: {e}")

    # If no recommendations from seed artists, try using age-specific genres
    if not recommendations or not recommendations['tracks']:
        try:
            recommendations = sp.recommendations(seed_genres=age_genres + [genre], limit=5, market=region, **mood_features)
        except Exception as e:
            st.write(f"Error fetching recommendations with seed genres: {e}")

    songs = []
    if recommendations and recommendations['tracks']:
        for track in recommendations['tracks']:
            song_data = {
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album_art': track['album']['images'][0]['url'],
                'url': track['external_urls']['spotify'],
                'preview': track['preview_url'],
                'popularity': track['popularity'],
                'release_date': track['album']['release_date']
            }
            songs.append(song_data)
    else:
        st.write("No recommendations available for the selected mood, language, and age.")

    return songs

# Streamlit application layout
st.title("Mood-Based Music Recommendation System using Spotify API 🎵🔊🎧")

# User input for feelings
if 'user_feelings' not in st.session_state:
    st.session_state.user_feelings = ""
if 'detected_mood' not in st.session_state:
    st.session_state.detected_mood = None

st.session_state.user_feelings = st.text_area("Describe your feelings:", height=150, value=st.session_state.user_feelings)

# User input for age
age = st.slider("Select your age:", 0, 100, 25)  # Default to 25


# After analyzing feelings and getting the detected mood
if st.button('Analyze Feelings'):
    with st.spinner('Analyzing your feelings...'):
        st.session_state.detected_mood = enhanced_analyze_feelings(st.session_state.user_feelings)

    # Display the speedometer for mood intensity
    if st.session_state.detected_mood:
        fig = create_speedometer(st.session_state.detected_mood)
        st.plotly_chart(fig)

# Define languages for recommendations
languages = {
    'English': 'US',
    'Hindi': 'IN',
    'Tamil': 'IN',
    'Telugu': 'IN',
    'Kannada': 'IN',
    'Malayalam': 'IN',
    'Bengali': 'IN',
    'Punjabi': 'IN',
    'Marathi': 'IN',
    'Gujarati': 'IN',
    'French': 'FR',
    'Spanish': 'ES',
    'German': 'DE',
    'Brazilian Portuguese': 'BR',
    'Japanese': 'JP'
}

selected_language = st.selectbox("Select language/market for recommendations:", list(languages.keys()))

# Check if mood is detected before getting recommendations
if st.session_state.detected_mood:
    region_code = languages[selected_language]
    recommendations = get_enhanced_recommendations(st.session_state.detected_mood, selected_language, region_code, age)

    st.write(f"### Recommended Songs for {st.session_state.detected_mood.capitalize()} mood in {selected_language}:")
    
    for song in recommendations:
        st.image(song['album_art'], width=200)
        st.write(f"{song['name']}** by {song['artist']}")
        st.write(f"[Listen on Spotify]({song['url']})")
        if song['preview']:
            st.audio(song['preview'])
        st.write(f"Popularity: {song['popularity']}, Released on: {song['release_date']}")
        st.write("---")

# Sidebar for additional features
st.sidebar.title("Additional Features")

# User feedback section
feedback = st.sidebar.text_area("Provide Feedback:", height=100)
if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.warning("Please enter your feedback before submitting.")

# User tips section
st.sidebar.subheader("Tips for Better Recommendations")
st.sidebar.write("1. Be descriptive about your feelings.")
st.sidebar.write("2. Use emotions or specific words to convey your mood.")
st.sidebar.write("3. You can always try again with different inputs for varied results.")

st.sidebar.subheader("For Voice Input")
st.sidebar.write("1.Click the text box ")
st.sidebar.write("2.Press Windows Key + H")
