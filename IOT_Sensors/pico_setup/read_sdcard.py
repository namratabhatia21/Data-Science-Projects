from machine import Pin, I2C, SPI
import utime as time
from dht import DHT, InvalidChecksum
from ds3231 import DS3231
import sdcard
import os

# Initialize the SPI for SD card
spi = SPI(0, baudrate=1000000, polarity=0, phase=0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
cs = Pin(17, Pin.OUT)

# Initialize SD card
sd = sdcard.SDCard(spi, cs)

# Mount SD card
vfs = os.VfsFat(sd)
os.mount(vfs, "/sd")

# Function to read data from SD card
def read_from_sd():
    try:
        with open('/sd/sensor_data.txt', 'r') as file:
            for line in file:
                print(line.strip())
    except Exception as e:
        print("Error reading from SD card:", e)

# Main script to read and print SD card contents
read_from_sd()


