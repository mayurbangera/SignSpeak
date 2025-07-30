from gtts import gTTS
from playsound import playsound
import os

text = "Hello, I am speaking now"
voice_file = "test_voice.mp3"

tts = gTTS(text=text, lang='en')
tts.save(voice_file)
playsound(voice_file)
os.remove(voice_file)
