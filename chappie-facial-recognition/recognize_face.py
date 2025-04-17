import cv2 as cv
import pygame
import argparse
import boto3
import time
import os
import speech_recognition as sr


def recognizeFace(client, image, collection):
    face_matched = False
    with open(image, 'rb') as file:
        response = client.search_faces_by_image(
            CollectionId=collection,
            Image={'Bytes': file.read()},
            MaxFaces=1,
            FaceMatchThreshold=85
        )
        if not response['FaceMatches']:
            face_matched = False
        else:
            face_matched = True
    return face_matched, response

def detectFace(frame, face_cascade):
    face_detected = False
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    if len(faces) > 0:
        face_detected = True
        timestr = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(directory, f"image_{timestr}.png")
        cv.imwrite(image_path, frame)
        print('Your image was saved to:', image_path)
    return face_detected, image_path if face_detected else None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Facial recognition')
    parser.add_argument('--collection', help='Collection Name', default='my-face-collection')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='/home/shreyas/Desktop/Smart-glasses/.venv/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    args = parser.parse_args()

    # Initialize OpenCV face detection
    face_cascade_name = args.face_cascade
    face_cascade = cv.CascadeClassifier(cv.samples.findFile(face_cascade_name))
    if face_cascade.empty():
        print('--(!)Error loading face cascade')
        return

    # Initialize AWS Rekognition client
    client = boto3.client('rekognition')
    
    #collection_id = 'my-face-collection'
    #response = client.create_collection(CollectionId=collection_id)

    # Initialize webcam
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print('--(!)Error opening video capture')
        return

    # Initialize polly for speech synthesis
    polly_client = boto3.client('polly')
    pygame.mixer.init()
    ret, frame = cam.read()
    if not ret:
        print('--(!) No captured frame -- Break!')
        os.abort()

    face_detected, image_path = detectFace(frame, face_cascade)

    if face_detected and image_path:
        face_matched, response = recognizeFace(client, image_path, args.collection)
        if face_matched:
            person_name = response['FaceMatches'][0]['Face']['ExternalImageId']
            similarity = round(response['FaceMatches'][0]['Similarity'], 1)
            confidence = round(response['FaceMatches'][0]['Face']['Confidence'], 2)
            print(f'Identity matched {person_name} with {similarity} similarity and {confidence} confidence.')
            response = polly_client.synthesize_speech(
                                        Engine='generative',
                                        Text=f'Person is {person_name}', 
                                        OutputFormat='mp3', 
                                        VoiceId='Kajal')
            with open('output.mp3', 'wb') as f:
                f.write(response['AudioStream'].read())
            pygame.mixer.music.load('output.mp3')
            pygame.mixer.music.play()
            if os.path.exists('output.mp3'):
                # Delete the file
                os.remove('output.mp3')
            print(f'Hello {person_name}! What is my purpose?')
        else:
            print('Unknown Human Detected!')
            pygame.mixer.music.load('resources/newPerson.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pass
            
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            print("BEEP!!")

            with mic as source:
                print("Listening for name...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=10)

            try:
                spoken_name = recognizer.recognize_google(audio)
                print(f"You said: {spoken_name}")
                spoken_name = spoken_name.strip().replace(" ", "_")
                # Save the unknown face to Rekognition with the spoken name
                with open(image_path, 'rb') as f:
                    index_response = client.index_faces(
                    CollectionId=args.collection,
                    Image={'Bytes': f.read()},
                    ExternalImageId=spoken_name,
                    DetectionAttributes=['ALL']
                )
    
                print(f"Added {spoken_name} to collection.")
    
                # Say thank you
                response = polly_client.synthesize_speech(
                    Engine='generative',
                    Text=f'Thank you {spoken_name}, I will remember you!', 
                    OutputFormat='mp3', 
                    VoiceId='Kajal'
                )
                with open('output.mp3', 'wb') as f:
                    f.write(response['AudioStream'].read())
                pygame.mixer.music.load('output.mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
                if os.path.exists('output.mp3'):
                # Delete the file
                    os.remove('output.mp3')

            except sr.UnknownValueError:
                print("Sorry, could not understand the name.")
                response = polly_client.synthesize_speech(
                    Engine='generative',
                    Text='Sorry, I could not understand the name. Please try again next time.', 
                    OutputFormat='mp3', 
                    VoiceId='Kajal'
                )
            with open('output.mp3', 'wb') as f:
                f.write(response['AudioStream'].read())
            pygame.mixer.music.load('output.mp3')
            pygame.mixer.music.play()
            #while pygame.mixer.music.get_busy():
                #pass
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)  # pause for 100 milliseconds
            if os.path.exists('output.mp3'):
                # Delete the file
                os.remove('output.mp3')
            
        time.sleep(10)
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
        print(f'Deleted temporary image: {image_path}')

    if cv.waitKey(20) & 0xFF == ord('q'):
        os.abort()

    # Release resources
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Create directory for saving images
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'faces')
    os.makedirs(directory, exist_ok=True)

    # Run the main function
    main()




