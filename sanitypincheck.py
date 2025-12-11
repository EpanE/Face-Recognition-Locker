from gpiozero import Device, OutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
import time

# Force gpiozero to use the LGPIO backend (works on Pi 5)
Device.pin_factory = LGPIOFactory()

relay = OutputDevice(
    17,               # BCM17 = physical pin 11
    active_high=False,  # typical active LOW relay module
    initial_value=True  # start HIGH → relay OFF at startup
)

print("Relay ON for 3 seconds")
relay.on()           # active_high=False → on() drives pin LOW → relay energised
time.sleep(3)
print("Relay OFF")
relay.off()
