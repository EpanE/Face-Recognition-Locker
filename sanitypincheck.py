from gpiozero import OutputDevice
import time

relay = OutputDevice(17, active_high=False, initial_value=True)

print("Relay ON for 3 seconds")
relay.on()
time.sleep(3)
print("Relay OFF")
relay.off()
