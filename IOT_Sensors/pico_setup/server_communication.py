import os
from machine import Pin, SPI
import network
import utime as time
import urequests

# Initialize WLAN interface
wlan = network.WLAN(network.STA_IF)

# Activate the WLAN interface
wlan.active(True)

# Connect to the specified Wi-Fi network
ssid = "La Marq-901"
wlan.connect(ssid)

# Check connection status and print network configuration
while not wlan.isconnected():
    time.sleep(1)  # Wait for connection

# Once connected, print the network configuration
status = wlan.ifconfig()
print("Wi-Fi connected successfully!")
print("IP Address:", status[0])
print("Subnet Mask:", status[1])
print("Gateway:", status[2])
print("DNS Server:", status[3])

# Initialize the SPI for SD card (assuming your setup is correct)
spi = SPI(0, baudrate=1000000, polarity=0, phase=0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
cs = Pin(17, Pin.OUT)

# Function to mount SD card
def mount_sd_card():
    try:
        import sdcard
        sd = sdcard.SDCard(spi, cs)
        vfs = os.VfsFat(sd)
        os.mount(vfs, "/sd")
        print("SD card mounted successfully.")
        return True
    except Exception as e:
        print("Error mounting SD card:", e)
        return False

# Function to check if SD card is available
def check_sd_card():
    if not mount_sd_card():
        print("SD card is not available.")
        return False
    print("SD card is available.")
    return True

# Function to read data from SD card
def read_data_from_sd(filename):
    try:
        if not check_sd_card():
            return []
        
        if filename not in os.listdir('/sd'):
            print(f"File '{filename}' not found on SD card.")
            return []

        with open('/sd/' + filename, 'r') as file:
            data = file.readlines()
        return [line.strip() for line in data]
    except Exception as e:
        print(f"Error reading from SD card '{filename}':", e)
        return []

# Function to send data to Flask API
def send_data_to_server(data):
    url = "http://bhatianamrata.pythonanywhere.com/api/post_data"  
    headers = {'Content-Type': 'application/json'}

    for entry in data:
        date_str, time_str, temperature, humidity = entry.split(", ")
        payload = {
            "temperature": temperature.split(": ")[1],
            "humidity": humidity.split(": ")[1],
            "date": date_str.split(": ")[1],
            "time": time_str.split(": ")[1]
        }
        try:
            response = urequests.post(url, json=payload, headers=headers)
            print("Server response:", response.status_code, response.text)
            response.close()
        except Exception as e:
            print("Error sending data:", e)
            return False  # Return False if any error occurs
    return True

# Function to clear SD card file
def clear_sd_file(filename):
    try:
        with open('/sd/' + filename, 'w') as file:
            pass  # Opens the file in write mode, effectively clearing its contents
        print(f"File '{filename}' cleared successfully.")
    except Exception as e:
        print(f"Error clearing file '{filename}': {e}")

# Example usage:
filename = 'sensor_data.txt'  

# Main loop to run every 5 minutes
while True:
    try:
        data = read_data_from_sd(filename)
        if wlan.isconnected() and data:
            if send_data_to_server(data):
                clear_sd_file(filename)
    except Exception as e:
        print("Error in main loop:", e)

    time.sleep(300)  # Sleep for 5 minutes (300 seconds) before running again
