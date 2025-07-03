import socket
import os

SERVER_IP = "localhost"  # Change to your local machine's IP
PORT = 12346

def send_image(client, image_name):
    """Sends an image file to the server."""
    file_size = os.path.getsize(image_name)
    print(f"Sending image: {image_name} ({file_size} bytes)")

    with open(image_name, "rb") as f:
        sent_size = 0
        while chunk := f.read(65536):
            client.sendall(chunk)
            sent_size += len(chunk)
    
    print(f"Image {image_name} sent successfully ({sent_size}/{file_size} bytes)")

def request_mask(image_name):
    """Sends an image file to the server and receives the drawn mask."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_IP, PORT))

    # Step 1: Send filename and expected file size
    print(f"image_name: {image_name}")
    file_size = os.path.getsize(image_name)
    header = f"{image_name}|{file_size}".encode()
    client.sendall(header)

    # Step 2: Wait for acknowledgment
    ack = client.recv(1024).decode()
    if ack != "FILENAME_RECEIVED":
        print("Error: Filename acknowledgment not received.")
        client.close()
        return

    # Step 3: Send image file
    send_image(client, image_name)

    # Step 4: Receive mask data
    mask_data = b""
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        mask_data += chunk

    # Step 5: Save the received mask
    mask_filename = f"{image_name.split('.png')[0]}_mask.png"
    with open(mask_filename, "wb") as f:
        f.write(mask_data)

    print(f"Received mask saved as {mask_filename}")
    client.close()
    return mask_filename


# request_mask(image_name="test.png")#flux-fill-dev.png")
