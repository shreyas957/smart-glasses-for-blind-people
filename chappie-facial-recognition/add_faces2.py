import cv2
import boto3
import time
import os
import speech_recognition as sr
import pygame

# Initialize AWS clients
polly_client = boto3.client('polly')
rekognition_client = boto3.client('rekognition')

# Initialize speech recognizer and microphone (Logitech BCC950)
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=3)

def speak_text(text):
    # Use AWS Polly to convert text to speech
    response = polly_client.synthesize_speech(
        Engine='generative',
        Text=text,
        OutputFormat='mp3',
        VoiceId='Kajal'
    )

    # Save the audio stream to a file
    audio_path = 'name_audio.mp3'
    with open(audio_path, 'wb') as file:
        file.write(response['AudioStream'].read())

    # Play the audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    # Clean up
    pygame.mixer.quit()
    os.remove(audio_path)

# Step 1: Take voice input for name
with mic as source:
    print("?? Please say the name of the person...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

try:
    name = recognizer.recognize_google(audio)
    name = name.strip().replace(" ", "_")
    print(f"? Detected name: {name}")
    speak_text(f"Detected name is {name}")
except sr.UnknownValueError:
    print("? Could not understand the name.")
    speak_text("Sorry, I could not understand the name.")
    exit()
except sr.RequestError as e:
    print(f"? Google Speech API error: {e}")
    speak_text("Google speech recognition failed.")
    exit()

# Step 2: Capture image from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("? Error: Unable to access the webcam.")
    speak_text("Unable to access the webcam.")
    exit()

ret, frame = cap.read()
dirname = os.path.dirname(__file__)
directory = os.path.join(dirname, 'faces')
os.makedirs(directory, exist_ok=True)
timestr = time.strftime("%Y%m%d-%H%M%S")
image_path = f'{directory}/image_{timestr}.jpg'
cv2.imwrite(image_path, frame)
cap.release()

print(f"?? Image saved at: {image_path}")
speak_text("Image captured successfully.")

# Step 3: Index the face in Rekognition
with open(image_path, 'rb') as file:
    response = rekognition_client.index_faces(
        Image={'Bytes': file.read()},
        CollectionId='chappie-faces',
        ExternalImageId=name,
        DetectionAttributes=['ALL']
    )

print("? Face indexed successfully.")
speak_text("Face indexed successfully.")
