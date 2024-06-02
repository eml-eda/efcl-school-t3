import time
import socket
import PySimpleGUI as sg
from pynput import keyboard

# flag to detect "Enter" was pressed.
enter_detected = False

# hander for the keypress
def on_press(key):
    global enter_detected
    try:
        if(key == keyboard.Key.enter):
            enter_detected = True
        #Add your code to drive motor
    except AttributeError:
        print("")

# waits until "Enter" is pressed while updating the GUI. 
def wait_enter():
    global enter_detected
    global window
    enter_detected = False
    while True:
        if enter_detected:
            break
        else:
            event, values = window.read(timeout=0)

    enter_detected = False

# Constants
IMAGE_FOLDER = ".//resources//resized//"
BASE_MOVEMENTS = ["open hand",
                  "fist (power grip)",
                  "index pointed",
                  "ok (thumb up)",
                  "right flexion (wrist supination)",
                  "left flexion (wristpronation)",
                  "horns",
                  "shaka",
                  ]
REPETITIONS = 5
GESTURE_TIME = 8  # seconds
REST_TIME = 5  # seconds

# Init movements and labels
movements = [mov for mov in BASE_MOVEMENTS for _ in range(REPETITIONS)]
label = list(range(1, 9))
label = sorted(label * REPETITIONS)
print(label)

# Init counters
i = 0
start_rest = 0
end_rest = 0
end = 1

# Create a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Ensure that you can restart your server quickly when it terminates
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Set the client socket's TCP "well-known port" number
well_known_port = 3333
size = 243
sock.bind(("127.0.0.1", 30000))
layout = [
    [sg.Text(text='EMG Acquisition Session',
             font=('Arial Bold', 16),
             size=20, expand_x=True,
             justification='center',
             background_color='#ffffff')],
    [sg.Image(f'{IMAGE_FOLDER}ins1.png',
              expand_x=True,
              expand_y=True,
              background_color='#ffffff',
              key='image')]
]


# Create the window
window = sg.Window('EMG Acquisition Session', layout, size=(800, 600), keep_on_top=True, background_color='#ffffff')

# Set the number of clients waiting for connection that can be queued
sock.listen(5)
event, values = window.read(timeout=0)

# create a listener for the keyboard
listener = keyboard.Listener(on_press=on_press)
listener.start()

# wait user input
stop = 1
wait_enter()

# display each gesture for reference
for i_ in range(1, 9):
    image = f'{IMAGE_FOLDER}{i_}.png'
    window['image'].update(image)
    event, values = window.read(timeout=0)
    wait_enter()

# display instruction 2
image = f'{IMAGE_FOLDER}ins2.png'
window['image'].update(image)
event, values = window.read(timeout=0)
wait_enter()

# display instruction 3
image = f'{IMAGE_FOLDER}ins3.png'
window['image'].update(image)
event, values = window.read(timeout=0)
wait_enter()

# display instruction 4
image = f'{IMAGE_FOLDER}ins4.png'
window['image'].update(image)
event, values = window.read(timeout=0)
wait_enter()

# display instruction 5 and wait for the TCP activation
image = f'{IMAGE_FOLDER}ins5.png'
window['image'].update(image)
event, values = window.read(timeout=0)
newSocket, address = sock.accept()
print("Connected from", address)

# display instruction 6
image = f'{IMAGE_FOLDER}ins6.png'
window['image'].update(image)
event, values = window.read(timeout=0)
wait_enter()

# display starting....
image = f'{IMAGE_FOLDER}starting.png'
window['image'].update(image)
event, values = window.read(timeout=0)

# main code
try:
    while 1:

        # loop serving the new client
        tic = time.perf_counter()
        start_mov = time.perf_counter()
        n = 0
        start_rest = time.perf_counter()

        while end:
            event, values = window.read(timeout=0)
            # number of byte received
            # receivedData = 1
            toc = time.perf_counter()
            end_mov = time.perf_counter()
            end_rest = time.perf_counter()
            n = n + 1
            if toc - tic > 1:
                tic = time.perf_counter()
                n = 0

            # When the `GESTURE_TIME` is reached, send the stop signal and change the image to `stop.png
            if end_mov - start_mov >= GESTURE_TIME and stop == 0:
                stop = 1
                print("stop")
                image = f'{IMAGE_FOLDER}0.png'
                window['image'].update(image)
                newSocket.sendall(b'\x00')
                start_rest = time.perf_counter()

            # When we ended the last movement, we stop the loop
            if i == len(movements) and stop == 1:
                print("end")
                end = 0
                break

            # When the `REST_TIME` is reached, send the start signal and change the image to the next movement
            if stop == 1 and end_rest - start_rest >= REST_TIME:
                stop = 0
                start_mov = time.perf_counter()
                image = f'{IMAGE_FOLDER}{label[i]}.png'
                window['image'].update(image)
                newSocket.sendall(label[i].to_bytes(2, 'little'))
                print(movements[i])
                i = i + 1

            # If the user closes the window, we stop the loop
            if event == sg.WIN_CLOSED:
                break

        newSocket.sendall(b'\x00')
        image = f'{IMAGE_FOLDER}endpos1.png'
        window['image'].update(image)
        event, values = window.read(timeout=0)
        wait_enter()

        image = f'{IMAGE_FOLDER}endpos2.png'
        window['image'].update(image)
        event, values = window.read(timeout=0)
        wait_enter()

        image = f'{IMAGE_FOLDER}endpos3.png'
        window['image'].update(image)
        event, values = window.read(timeout=0)
        wait_enter()

        break


except KeyboardInterrupt:
    print("exit")

finally:
    print("end")
