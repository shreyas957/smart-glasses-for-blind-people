import cv2 as cv
import pygame
import argparse
import boto3
import time
import os


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
            response = polly_client.synthesize_speech(Text=f'Person is {person_name}', 
                                        OutputFormat='mp3', 
                                        VoiceId='Joanna')
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
            response = polly_client.synthesize_speech(Text='Unknown Person', 
                                        OutputFormat='mp3', 
                                        VoiceId='Joanna')
            with open('output.mp3', 'wb') as f:
                f.write(response['AudioStream'].read())
                                        
            pygame.mixer.music.load('output.mp3')
            pygame.mixer.music.play()
            if os.path.exists(f'output.mp3'):
                # Delete the file
                os.remove('output.mp3')
        time.sleep(10)

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




