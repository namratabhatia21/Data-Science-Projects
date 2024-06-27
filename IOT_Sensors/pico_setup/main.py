from machine import Pin, I2C, SPI
import utime as time
from dht import DHT, InvalidChecksum
from ds3231 import DS3231
import sdcard
import os
import urequests
from ds3231 import *
import time


# Initialize the DHT sensor
dht_pin = Pin(28, Pin.OUT, Pin.PULL_DOWN)
sensor = DHT(dht_pin)

# Initialize the I2C for DS3231 with GPIO 5 for SCL and GPIO 4 for SDA
sda_pin = Pin(4)
scl_pin = Pin(5)
i2c = I2C(0, scl=scl_pin, sda=sda_pin)  # Use I2C0

# Initialize the DS3231 RTC
ds = DS3231(i2c)
time.sleep(0.5)  # Allow some time for the RTC to initialize

# Initialize the LED
led = Pin(1, Pin.OUT)

# Initialize the SPI for SD card
spi = SPI(0, baudrate=1000000, polarity=0, phase=0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
cs = Pin(17, Pin.OUT)
# Set the DS3231 RTC to current system time
ds.set_time()

# Initialize SD card
sd = sdcard.SDCard(spi, cs)

# Mount SD card
vfs = os.VfsFat(sd)
os.mount(vfs, "/sd")

# Function to blink the LED
def blink_led(duration_ms):
    led.value(1)
    time.sleep_ms(duration_ms)
    led.value(0)

# Function to log data to SD card
def log_to_sd(data):
    try:
        with open('/sd/sensor_data.txt', 'a') as file:
            file.write(data + '\n')
    except Exception as e:
        print("Error writing to SD card:", e)

# Main loop
while True:
    time.sleep(5)  # Wait for 5 seconds before taking the next reading

    try:
        # Read temperature and humidity from the DHT sensor
        temperature = sensor.temperature
        humidity = sensor.humidity

        # Get the current date and time from the DS3231 RTC
        current_time = ds.get_time()
        date_str = "{}/{}/{}".format(current_time[1], current_time[2], current_time[0])
        time_str = "{}:{}:{}".format(current_time[3], current_time[4], current_time[5])

        # Print the readings along with the date and time
        print("Date: {}".format(date_str))
        print("Time: {}".format(time_str))
        print("Temperature: {} C".format(temperature))
        print("Humidity: {} %".format(humidity))

        # Blink the LED
        blink_led(200)  # Blink for 200 ms

        # Prepare the data string
        data = "Date: {}, Time: {}, Temperature: {} C, Humidity: {} %".format(date_str, time_str, temperature, humidity)
        

        # Log the data to SD card
        log_to_sd(data)

    except InvalidChecksum:
        print("Checksum error with DHT sensor")
    except Exception as e:
        print("Error: {}".format(e))


