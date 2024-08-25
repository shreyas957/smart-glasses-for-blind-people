import RPi.GPIO as GPIO
import subprocess
import time

button1 = 13
button2 = 15
button3 = 16

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(button1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(button2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(button3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print('     1. Object detection \n     2. Text to speech \n     3.face recognize')

def check_buttons():
    while True:
        # Check state of each button
        if GPIO.input(button1) == GPIO.LOW:
            print('Button 1 pressed : Object detection...')
            subprocess.run(["python", "object_detection.py"])
            print('DONE')
            # Wait for button release
            while GPIO.input(button1) == GPIO.LOW:
                time.sleep(0.1)  # Poll every 0.1 seconds to check button release

        if GPIO.input(button2) == GPIO.LOW:
            print('Button 2 pressed. : Text_to_speech..')
            subprocess.run(["python", "text_to_speech.py"])
            print('DONE')
            # Wait for button release
            while GPIO.input(button2) == GPIO.LOW:
                time.sleep(0.1)  # Poll every 0.1 seconds to check button release

        if GPIO.input(button3) == GPIO.LOW:
            print('Button 3 pressed. : face recognition..')
            subprocess.run(["python", "recognize_face.py"])
            print('DONE')
            # Wait for button release
            while GPIO.input(button3) == GPIO.LOW:
                time.sleep(0.1)  # Poll every 0.1 seconds to check button release

        time.sleep(0.1)  # Poll buttons every 0.1 seconds to check for presses

def end_program():
    GPIO.cleanup()

if __name__ == '__main__':
    setup()

    try:
        check_buttons()

    except KeyboardInterrupt:
        print('Keyboard interrupt detected.')

    finally:
        end_program()
