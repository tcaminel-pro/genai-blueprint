import logging
import sys
import wave

import pyaudio


def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == "darwin" else 2
RATE = 44100
RECORD_SECONDS = 5

try:
    with wave.open("output.wav", "wb") as wf:
        p = pyaudio.PyAudio()

        # Check if audio input is available
        try:
            device_count = p.get_device_count()
            input_devices = []

            for i in range(device_count):
                info = p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    input_devices.append(i)

            if not input_devices:
                raise RuntimeError("No audio input devices found")

            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)

            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_devices[0] if input_devices else None,
            )

            print("Recording...")
            for _ in range(RATE // CHUNK * RECORD_SECONDS):
                wf.writeframes(stream.read(CHUNK))
            print("Done")

            stream.close()

        except Exception as e:
            print(f"Audio recording error: {e}")
            print("Creating a silent/dummy audio file instead...")

            # Create a silent audio file as fallback
            silent_frames = b"\x00" * CHUNK
            for _ in range(RATE // CHUNK * RECORD_SECONDS):
                wf.writeframes(silent_frames)

        finally:
            p.terminate()

except Exception as e:
    logging.error(f"Fatal error: {e}")
    print("This script requires audio hardware. For development/testing, consider using:")
    print("1. A system with microphone/line-in")
    print("2. Virtual audio devices")
    print("3. Mock audio data for testing")

    print("*** known audio devices:")

    list_audio_devices()
