import socket
import cv2
import numpy as np
import os

HOST = "0.0.0.0"
PORT = 12345

def receive_image(conn, image_name, file_size):
    """Receives an image file from the remote client."""
    received_size = 0
    with open(image_name, "wb") as f:
        while received_size < file_size:
            data = conn.recv(min(file_size - received_size, 65536))  # Read remaining size
            if not data:
                break
            f.write(data)
            received_size += len(data)

    print(f"Image {image_name} received ({received_size}/{file_size} bytes)")

    if received_size < file_size:
        print("Warning: File transfer incomplete!")

def draw_mask(image_path):
    """Allows the user to draw a mask on the given image and saves the mask."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

    def draw(event, x, y, flags, param):
        """Callback function to handle drawing on the mask."""
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE] and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(mask, (x, y), 20, (255, 255, 255), -1)  # Bolder strokes

    cv2.namedWindow("Draw Mask")
    cv2.setMouseCallback("Draw Mask", draw)

    while True:
        overlay = cv2.addWeighted(image, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
        cv2.imshow("Draw Mask", overlay)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    mask_filename = f"{os.path.splitext(image_path)[0]}_mask.png"
    cv2.imwrite(mask_filename, mask)
    cv2.destroyAllWindows()
    return mask_filename

def start_server():
    """Listens for incoming connections, receives an image, and processes it."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Server listening on {HOST}:{PORT}...")

    while True:
        conn, addr = server.accept()
        print(f"Received connection from {addr}")

        # Step 1: Receive filename and expected file size
        header = conn.recv(1024).decode().strip()
        image_name, file_size = header.split("|")
        file_size = int(file_size)
        image_name = os.path.basename(image_name)  # Extract only the file name
        print(f"Receiving image: {image_name} (Expected size: {file_size} bytes)")

        # Step 2: Acknowledge receipt of the filename and file size
        conn.sendall(b"FILENAME_RECEIVED")

        # Step 3: Receive the image file
        receive_image(conn, image_name, file_size)

        # Step 4: Verify file exists and is readable
        if not os.path.exists(image_name):
            print(f"Error: File {image_name} was not saved properly!")
            conn.close()
            continue

        img = cv2.imread(image_name)
        if img is None:
            print(f"Error: OpenCV failed to load {image_name}")
            conn.close()
            continue

        # Step 5: Draw the mask
        mask_path = draw_mask(image_name)
        if mask_path and os.path.exists(mask_path):
            with open(mask_path, "rb") as f:
                while chunk := f.read(65536):
                    conn.sendall(chunk)
            
            print(f"Sent mask file {mask_path} to {addr}")

        conn.close()

start_server()
