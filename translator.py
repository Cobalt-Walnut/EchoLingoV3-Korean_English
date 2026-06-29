'''
This code was finalized on 29 Jun, 2026 by reconstructing snippets of code that is on a plane to South Korea
for the FIRST LEGO League Korea Open Invitational with FLL team 66538 - TheVikings TempleTech :D

It's a bidirectional English to Korean (and vice versa) translation program that is running on a Raspberry Pi 5
with a MAX98357A amp, a 3ohm 5w speaker, a USB mic, 2 buttons, 2 LEDs, and a SPDT toggle switch. The case is 
custom and can be found on GitHub at github.com/Cobalt-Walnut/EchoLingoV3-Korean_English/CAD
'''

from gpiozero import LED, Button, DigitalInputDevice
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
import os
import sys
import re
import tempfile
import threading
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    VitsModel,
)

# ---------------------------------------------------------------------------
# Audio Devices
# ---------------------------------------------------------------------------
APLAY_DEVICE = "plughw:MAX98357A"
INPUT_DEVICE_NAME = "USB PnP Sound Device"        # USB Mic (PnP Sound Device)
RECORD_RATE  = 44100    # native USB mic rate
WHISPER_RATE = 16000    # whisper requires 16000Hz
SAMPLE_RATE  = WHISPER_RATE

# ---------------------------------------------------------------------------
# GPIO
# ---------------------------------------------------------------------------
RED_LED_PIN   = 22
GREEN_LED_PIN = 27
BUTTON_PIN    = 4
SWITCH_PIN    = 13
EXIT_BTN_PIN  = 5

red_led          = LED(RED_LED_PIN)
green_led        = LED(GREEN_LED_PIN)
record_button    = Button(BUTTON_PIN,    pull_up=True)
direction_switch = DigitalInputDevice(SWITCH_PIN, pull_up=False)
exit_button      = Button(EXIT_BTN_PIN, pull_up=True, bounce_time=0.1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WHISPER_BIN   = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-base.bin")
PIPER_BIN     = "/home/echolingo/translator-env/bin/piper"
PIPER_MODEL   = os.path.expanduser("~/piper_models/en_US-ryan-high.onnx")
PIPER_CONFIG  = os.path.expanduser("~/piper_models/en_US-ryan-high.onnx.json")
SOUNDS_DIR    = os.path.expanduser("~/sounds")
PISUGAR_SOCKET = "/tmp/pisugar-server.sock"

# Sound files
SND_STARTUP          = os.path.join(SOUNDS_DIR, "startup.wav")
SND_MODE_KO_EN       = os.path.join(SOUNDS_DIR, "mode_ko_en.wav")
SND_MODE_EN_KO       = os.path.join(SOUNDS_DIR, "mode_en_ko.wav")
SND_NO_SPEECH        = os.path.join(SOUNDS_DIR, "no_speech.wav")
SND_GOODBYE          = os.path.join(SOUNDS_DIR, "goodbye.wav")
SND_BATTERY_30       = os.path.join(SOUNDS_DIR, "battery_30.wav")
SND_BATTERY_25       = os.path.join(SOUNDS_DIR, "battery_25.wav")
SND_BATTERY_20       = os.path.join(SOUNDS_DIR, "battery_20.wav")
SND_BATTERY_CRITICAL = os.path.join(SOUNDS_DIR, "battery_critical.wav")
SND_STARTUP_BLIP     = os.path.join(SOUNDS_DIR, "startup_blip.wav")
SND_BLIP             = os.path.join(SOUNDS_DIR, "blip.wav")

# ---------------------------------------------------------------------------
# Protected competition terms
# ---------------------------------------------------------------------------
PROTECTED_TERMS = [
    "FIRST LEGO League",
    "SPIKE Prime",
    "SPIKE",
    "EV3",
    "FLL",
    "FIRST",
    "LEGO",
]

WHISPER_PROMPT = (
    "FIRST LEGO League, FLL, FIRST, LEGO, EV3, SPIKE, SPIKE Prime, "
    "Robot Game, Innovation Project, Core Values"
)
# ---------------------------------------------------------------------------
# TTS settings
# ---------------------------------------------------------------------------
TTS_VOLUME        = "0.70"
PIPER_SPEED       = "1.4"   # length_scale: >1.0 = slower, <1.0 = faster
MMS_SPEAKING_RATE = 0.6    # <1.0 = slower for MMS-TTS

# ---------------------------------------------------------------------------
# Exit flag
# ---------------------------------------------------------------------------
exit_requested = False

# ---------------------------------------------------------------------------
# Battery warning state
# ---------------------------------------------------------------------------
battery_warnings_fired = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_input_device():
    for i, dev in enumerate(sd.query_devices()):
        if (
            INPUT_DEVICE_NAME.lower() in dev["name"].lower()
            and dev["max_input_channels"] > 0
        ):
            return i

    raise RuntimeError(
        f"Could not find input device: {INPUT_DEVICE_NAME}"
    )
    
def play(path):
    if os.path.exists(path):
        subprocess.run(["aplay", "-D", APLAY_DEVICE, path])
    else:
        print(f"[WARN] Sound file missing: {path}")

def play_tts_wav(tmp_wav):
    reduced = tmp_wav + ".reduced.wav"
    subprocess.run(["sox", tmp_wav, reduced, "vol", TTS_VOLUME], check=True)
    subprocess.run(["aplay", "-D", APLAY_DEVICE, reduced])
    os.unlink(reduced)

def get_direction():
    return "ko_to_en" if direction_switch.value == 1 else "en_to_ko"

def direction_label(d):
    return "KO->EN" if d == "ko_to_en" else "EN->KO"

# ---------------------------------------------------------------------------
# Exit handler
# ---------------------------------------------------------------------------
def exit_program():
    global exit_requested
    print("\nExit button pressed.")
    for _ in range(8):
        red_led.toggle()
        time.sleep(0.25)
    red_led.off()
    green_led.off()
    play(SND_GOODBYE)
    exit_requested = True

exit_button.when_pressed = exit_program

# ---------------------------------------------------------------------------
# PiSugar 3 Plus — battery via Unix socket
# ---------------------------------------------------------------------------
def get_battery_level():
    if not os.path.exists(PISUGAR_SOCKET):
        return None
    try:
        result = subprocess.run(
            ["nc", "-q", "0", "-U", PISUGAR_SOCKET],
            input="get battery",
            capture_output=True,
            text=True,
            timeout=3
        )
        for line in result.stdout.splitlines():
            if line.startswith("battery:"):
                return int(float(line.split(":")[1].strip()))
    except Exception as e:
        print(f"[WARN] Battery read failed: {e}")
    return None

def generate_battery_startup_wav(level):
    text    = f"Battery at {level} percent."
    tmp_raw = os.path.join(SOUNDS_DIR, "battery_startup.raw.wav")
    tmp     = os.path.join(SOUNDS_DIR, "battery_startup.wav")
    proc = subprocess.Popen(
        [PIPER_BIN,
         "--model",        PIPER_MODEL,
         "--config",       PIPER_CONFIG,
         "--length-scale", PIPER_SPEED,
         "--output_file",  tmp_raw],
        stdin=subprocess.PIPE
    )
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()
    proc.wait()
    subprocess.run(["sox", tmp_raw, tmp, "vol", TTS_VOLUME], check=True)
    os.unlink(tmp_raw)
    return tmp

def check_battery_warnings(level):
    global exit_requested
    if level <= 15 and "critical" not in battery_warnings_fired:
        battery_warnings_fired.add("critical")
        print(f"[BATTERY] Critical: {level}%")
        red_led.on()
        green_led.on()
        play(SND_BATTERY_CRITICAL)
        red_led.off()
        green_led.off()
        exit_requested = True
        return
    if level <= 20 and "20" not in battery_warnings_fired:
        battery_warnings_fired.add("20")
        print(f"[BATTERY] Warning 20%: {level}%")
        play(SND_BATTERY_20)
    elif level <= 25 and "25" not in battery_warnings_fired:
        battery_warnings_fired.add("25")
        print(f"[BATTERY] Warning 25%: {level}%")
        play(SND_BATTERY_25)
    elif level <= 30 and "30" not in battery_warnings_fired:
        battery_warnings_fired.add("30")
        print(f"[BATTERY] Warning 30%: {level}%")
        play(SND_BATTERY_30)

def battery_monitor_loop():
    """Poll battery every 5 minutes in background thread."""
    while not exit_requested:
        time.sleep(300)   # wait first, startup already checked it
        if exit_requested:
            break
        level = get_battery_level()
        if level is not None:
            print(f"[BATTERY] {level}%")
            check_battery_warnings(level)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading NLLB-200 translation model...")
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model.eval()

print("Loading MMS-TTS Korean model...")
mms_tokenizer  = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
mms_model      = VitsModel.from_pretrained("facebook/mms-tts-kor")
mms_model.eval()
KO_SAMPLE_RATE = mms_model.config.sampling_rate

print("All models loaded.\n")

# ---------------------------------------------------------------------------
# Term protection
# ---------------------------------------------------------------------------
def protect_terms(text):
    placeholders = {}
    out = text
    for i, term in enumerate(sorted(PROTECTED_TERMS, key=len, reverse=True)):
        placeholder = f"<TERM{i}>"
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(out):
            placeholders[placeholder] = term
            out = pattern.sub(placeholder, out)
    return out, placeholders

def restore_terms(text, placeholders):
    out = text
    for placeholder, term in placeholders.items():
        out = out.replace(placeholder, term)
        plain = placeholder.replace("<", "").replace(">", "")
        out = out.replace(plain, term)

    return out

ABBREVIATIONS = {
    "f l l": "FLL",
    "fll": "FLL",

    "first lego league": "FIRST LEGO League",

    "first": "FIRST",
    "lego": "LEGO",

    "ev three": "EV3",
    "ev-3": "EV3",
    "ev 3": "EV3",

    "spike": "SPIKE",
    "spike prime": "SPIKE Prime",
}

def normalize_abbreviations(text):
    out = text

    for spoken, official in ABBREVIATIONS.items():
        pattern = re.compile(r"\b" + re.escape(spoken) + r"\b", re.IGNORECASE)
        out = pattern.sub(official, out)

    return out
# ---------------------------------------------------------------------------
# STT — Whisper.cpp
# ---------------------------------------------------------------------------
def transcribe(audio_data, language):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sf.write(tmp_wav, audio_data, WHISPER_RATE)
    subprocess.run(
        [WHISPER_BIN,
         "-m", WHISPER_MODEL,
         "-l", language,
         "-otxt",
         "--no-timestamps",
         "--prompt", WHISPER_PROMPT,
         "-f", tmp_wav],
        capture_output=True, text=True
    )
    txt_file = tmp_wav + ".txt"
    text = ""
    if os.path.exists(txt_file):
        text = open(txt_file).read().strip()
        os.unlink(txt_file)
    os.unlink(tmp_wav)
    return text

# ---------------------------------------------------------------------------
# Translation — NLLB-200
# ---------------------------------------------------------------------------
def translate(text, tgt_lang_token):
    protected, placeholders = protect_terms(text)
    inputs     = nllb_tokenizer(protected, return_tensors="pt")
    forced_bos = nllb_tokenizer.convert_tokens_to_ids(tgt_lang_token)
    print("Protected :", protected)
    print("Placeholders:", placeholders)
    with torch.no_grad():
        output = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256
        )
    translated = nllb_tokenizer.decode(output[0], skip_special_tokens=True)
    return restore_terms(translated, placeholders)

# ---------------------------------------------------------------------------
# TTS — Piper (English) with speed control
# ---------------------------------------------------------------------------
ENGLISH_TTS_REPLACEMENTS = {
    "FIRST LEGO League": "FIRST LEGO League, ",
    "SPIKE Prime": "Spike Prime, ",
    "SPIKE": "Spike, ",
    "EV3": "E V Three, ",
    "FLL": "F L L, ",
}

def generate_english_tts(text):

    for original, spoken in sorted(
        ENGLISH_TTS_REPLACEMENTS.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        text = text.replace(original, spoken)

    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ).name

    proc = subprocess.Popen(
        [PIPER_BIN,
         "--model", PIPER_MODEL,
         "--config", PIPER_CONFIG,
         "--length-scale", PIPER_SPEED,
         "--output_file", tmp_wav],
        stdin=subprocess.PIPE
    )

    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()
    proc.wait()

    reduced = tmp_wav + ".reduced.wav"
    subprocess.run(
        ["sox", tmp_wav, reduced, "vol", TTS_VOLUME],
        check=True
    )

    os.unlink(tmp_wav)
    print(f"English TTS: {text}")
    return reduced

# ---------------------------------------------------------------------------
# TTS — MMS-TTS (Korean) with speed control
# ---------------------------------------------------------------------------
KOREAN_TTS_REPLACEMENTS = {
    "FLL": "에프 엘 엘. ",
    "EV3": "이브이 쓰리. ",
    "SPIKE": "스파이크. ",
    "SPIKE Prime": "스파이크 프라임. ",
    "FIRST LEGO League": "퍼스트 레고 리그. ",
}

def generate_korean_tts(text):

    for original, spoken in sorted(
        KOREAN_TTS_REPLACEMENTS.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        text = text.replace(original, spoken)
    inputs = mms_tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] < 20:
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, 20)
        )

    with torch.no_grad():
        output = mms_model(
            **inputs,
            speaking_rate=MMS_SPEAKING_RATE
        ).waveform

    audio = output.squeeze().numpy()
    audio = (audio * 32767).astype(np.int16)

    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    sf.write(tmp_wav, audio, KO_SAMPLE_RATE, subtype="PCM_16")

    reduced = tmp_wav + ".reduced.wav"

    subprocess.run(
        ["sox", tmp_wav, reduced, "vol", TTS_VOLUME],
        check=True
    )

    os.unlink(tmp_wav)
    print(f"Korean TTS: {text}")
    return reduced


# ---------------------------------------------------------------------------
# Play the TTS output
# ---------------------------------------------------------------------------
def play_generated_tts(path):
    subprocess.run(["aplay", "-D", APLAY_DEVICE, path])
    os.unlink(path)

# ---------------------------------------------------------------------------
# Recording — blip plays BEFORE mic opens
# ---------------------------------------------------------------------------
def record_while_held(current_direction):
    # Play blip BEFORE opening mic so it isn't recorded
    play(SND_STARTUP_BLIP)
    green_led.off()
    red_led.on()

    q         = queue.Queue()
    recording = True

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if recording:
            q.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=RECORD_RATE,
            channels=1,
            dtype="int16",
            device=get_input_device(),
            callback=callback
        )
        stream.start()
    except Exception as e:
        print(f"Failed to open input stream: {e}")
        red_led.off()
        return np.array([], dtype="int16"), None

    print("Recording... (release button to stop)")

    new_dir = None
    while record_button.is_active and not exit_requested:
        detected = get_direction()
        if detected != current_direction:
            new_dir = detected
            break
        time.sleep(0.01)

    recording = False
    stream.stop()
    stream.close()
    red_led.off()

    frames = []
    while not q.empty():
        frames.append(q.get())

    audio = np.concatenate(frames, axis=0) if frames else np.array([], dtype="int16")

    # Resample from 44100 to 16000 for Whisper using sox
    if audio.size > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_44k = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_16k = f.name
        sf.write(tmp_44k, audio, RECORD_RATE)
        subprocess.run(
            ["sox", tmp_44k, "-r", str(WHISPER_RATE), tmp_16k],
            capture_output=True
        )
        audio, _ = sf.read(tmp_16k, dtype="int16")
        os.unlink(tmp_44k)
        os.unlink(tmp_16k)

    return audio, new_dir

# ---------------------------------------------------------------------------
# Red LED blink during processing
# ---------------------------------------------------------------------------
def blink_red_while(flag_holder, interval=0.5):
    while flag_holder[0]:
        red_led.on()
        time.sleep(interval)
        red_led.off()
        time.sleep(interval)
    red_led.off()

# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------
def wait_for_button_or_switch(current_direction):
    while not exit_requested:
        if record_button.is_active:
            return "record"
        if get_direction() != current_direction:
            return "switch"
        time.sleep(0.01)
    return "exit"

def wait_for_playback_or_switch(current_direction):
    while not exit_requested:
        if record_button.is_active:
            return "play"
        if get_direction() != current_direction:
            return "switch"
        time.sleep(0.01)
    return "exit"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try:
    # Startup battery announcement
    batt = get_battery_level()
    if batt is not None:
        batt_wav = generate_battery_startup_wav(batt)
        play(batt_wav)
        check_battery_warnings(batt)

    play(SND_STARTUP)

    # Start battery monitor thread
    batt_thread = threading.Thread(target=battery_monitor_loop, daemon=True)
    batt_thread.start()

    current_direction = get_direction()
    print(f"Mode: {direction_label(current_direction)}")
    print("Hold button to record. Toggle switch to change direction.\n")

    while not exit_requested:
        red_led.off()
        green_led.on()

        print(f"[{direction_label(current_direction)}] Waiting...")
        action = wait_for_button_or_switch(current_direction)

        if action == "exit" or exit_requested:
            break

        if action == "switch":
            current_direction = get_direction()
            print(f"Switched to {direction_label(current_direction)}")
            play(SND_MODE_KO_EN if current_direction == "ko_to_en" else SND_MODE_EN_KO)
            continue

        # --- Record ---
        audio, new_dir = record_while_held(current_direction)

        if exit_requested:
            break

        if new_dir:
            current_direction = new_dir
            print(f"Switched to {direction_label(current_direction)}")
            play(SND_MODE_KO_EN if current_direction == "ko_to_en" else SND_MODE_EN_KO)
            continue

        if audio.size == 0:
            print("No audio captured.")
            play(SND_NO_SPEECH)
            continue

        # --- Transcribe + Translate with blinking red LED ---
        processing = [True]
        blink_thread = threading.Thread(
            target=blink_red_while, args=(processing,), daemon=True
        )
        blink_thread.start()

        print("Transcribing...")
        lang = "ko" if current_direction == "ko_to_en" else "en"

        text = transcribe(audio, lang)

        # Normalize competition terms
        text = normalize_abbreviations(text)

        print(f"Recognized : {text}")

        if not text.strip():
            processing[0] = False
            blink_thread.join()
            red_led.off()
            print("No speech detected.")
            play(SND_NO_SPEECH)
            continue

        print("Translating...")
        if current_direction == "ko_to_en":
            translation = translate(text, "eng_Latn")
        else:
            translation = translate(text, "kor_Hang")


        print(f"Translation: {translation}")
        print("Generating speech...")

        if current_direction == "ko_to_en":
            speech_file = generate_english_tts(translation)
        else:
            speech_file = generate_korean_tts(translation)

        processing[0] = False
        blink_thread.join()
        red_led.off()
        # --- Ready: green solid + blip ---
        green_led.on()
        play(SND_BLIP)
        print("Press button to play translation...")
        action = wait_for_playback_or_switch(current_direction)
        green_led.off()

        if action == "exit" or exit_requested:
            break

        if action == "switch":
            current_direction = get_direction()
            print(f"Switched to {direction_label(current_direction)}")
            play(SND_MODE_KO_EN if current_direction == "ko_to_en" else SND_MODE_EN_KO)
            continue

        # --- Speak ---
        print("Speaking...")
        play_generated_tts(speech_file)
        green_led.on()
        red_led.off()

except KeyboardInterrupt:
    print("\nKeyboard interrupt received.")
    red_led.off()
    green_led.off()
    os._exit(0)

finally:
    if exit_requested:
        print("Shutting down...")
        red_led.off()
        green_led.off()
        subprocess.run(
            ["nc", "-q", "0", "-U", PISUGAR_SOCKET],
            input="set shutdown",
            capture_output=True,
            text=True,
            timeout=3
        )
        os.system("sudo shutdown -h now")
