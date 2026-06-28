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
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    VitsModel,
)
from deepmultilingualpunctuation import PunctuationModel

# ---------------------------------------------------------------------------
# Audio Devices
# ---------------------------------------------------------------------------
APLAY_DEVICE  = "plughw:sndrpihifiberry"   # MAX98357 I2S amp
INPUT_DEVICE  = "plughw:0,0"               # USB mic
SAMPLE_RATE   = 16000

# ---------------------------------------------------------------------------
# GPIO
# ---------------------------------------------------------------------------
RED_LED_PIN    = 22
GREEN_LED_PIN  = 27
BUTTON_PIN     = 4
SWITCH_PIN     = 13
EXIT_BTN_PIN   = 5

red_led          = LED(RED_LED_PIN)
green_led        = LED(GREEN_LED_PIN)
record_button    = Button(BUTTON_PIN,   pull_up=True)
direction_switch = DigitalInputDevice(SWITCH_PIN, pull_up=False)
exit_button      = Button(EXIT_BTN_PIN, pull_up=True, bounce_time=0.1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WHISPER_BIN    = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL  = os.path.expanduser("~/whisper.cpp/models/ggml-base.bin")
PIPER_BIN      = "piper"   # installed into venv, available on PATH
PIPER_MODEL    = os.path.expanduser("~/piper_models/en_US-ryan-high.onnx")
PIPER_CONFIG   = os.path.expanduser("~/piper_models/en_US-ryan-high.onnx.json")

# ---------------------------------------------------------------------------
# Protected competition terms (passed to Whisper + shielded from NLLB)
# ---------------------------------------------------------------------------
PROTECTED_TERMS  = ["FLL", "FIRST", "SPIKE", "EV3", "LEGO"]
WHISPER_PROMPT   = ", ".join(PROTECTED_TERMS)

# ---------------------------------------------------------------------------
# Exit flag
# ---------------------------------------------------------------------------
exit_requested = False

def exit_program():
    global exit_requested
    print("\nExit button pressed.")
    for _ in range(8):
        red_led.toggle()
        time.sleep(0.25)
    subprocess.run(["aplay", "-D", APLAY_DEVICE,
                    os.path.expanduser("~/sounds/ev3_shutdown.wav")])
    exit_requested = True

exit_button.when_pressed = exit_program

# ---------------------------------------------------------------------------
# Load models at startup (all cached locally -- fully offline)
# ---------------------------------------------------------------------------
print("Loading punctuation model...")
punct_model = PunctuationModel(model="kredor/punctuate-all")

print("Loading NLLB-200 translation model...")
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model.eval()

print("Loading MMS-TTS Korean model...")
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
mms_model     = VitsModel.from_pretrained("facebook/mms-tts-kor")
mms_model.eval()
KO_SAMPLE_RATE = mms_model.config.sampling_rate

print("All models loaded. Ready.\n")

# ---------------------------------------------------------------------------
# Term protection helpers
# ---------------------------------------------------------------------------
def protect_terms(text):
    """Swap protected terms for placeholders before translation."""
    placeholders = {}
    out = text
    for term in PROTECTED_TERMS:
        placeholder = f"__{term}__"
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(out):
            placeholders[placeholder] = term
            out = pattern.sub(placeholder, out)
    return out, placeholders

def restore_terms(text, placeholders):
    """Put protected terms back after translation."""
    out = text
    for placeholder, term in placeholders.items():
        out = out.replace(placeholder, term)
    return out

# ---------------------------------------------------------------------------
# STT -- Whisper.cpp
# ---------------------------------------------------------------------------
def transcribe(audio_data, language):
    """Write audio to a temp WAV, run whisper-cli, return transcript."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    sf.write(tmp_wav, audio_data, SAMPLE_RATE)

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
# Translation -- NLLB-200
# ---------------------------------------------------------------------------
def translate(text, src_lang_token, tgt_lang_token):
    """Translate text between NLLB language tokens, protecting competition terms."""
    protected, placeholders = protect_terms(text)

    inputs     = nllb_tokenizer(protected, return_tensors="pt")
    forced_bos = nllb_tokenizer.convert_tokens_to_ids(tgt_lang_token)

    with torch.no_grad():
        output = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=256
        )

    translated = nllb_tokenizer.decode(output[0], skip_special_tokens=True)
    return restore_terms(translated, placeholders)

# ---------------------------------------------------------------------------
# TTS -- Piper (English)
# ---------------------------------------------------------------------------
def speak_english(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    proc = subprocess.Popen(
        [PIPER_BIN,
         "--model",  PIPER_MODEL,
         "--config", PIPER_CONFIG,
         "--output_file", tmp_wav],
        stdin=subprocess.PIPE
    )
    proc.stdin.write(text.encode("utf-8"))
    proc.stdin.close()
    proc.wait()

    subprocess.run(["aplay", "-D", APLAY_DEVICE, tmp_wav])
    os.unlink(tmp_wav)

# ---------------------------------------------------------------------------
# TTS -- MMS-TTS (Korean)
# ---------------------------------------------------------------------------
def speak_korean(text):
    inputs = mms_tokenizer(text, return_tensors="pt")

    # Pad short sequences to avoid narrow() bug
    if inputs["input_ids"].shape[1] < 10:
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, 10)
        )

    with torch.no_grad():
        output = mms_model(**inputs).waveform

    # Convert float32 to int16 for clean aplay output
    audio = output.squeeze().numpy()
    audio = (audio * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    sf.write(tmp_wav, audio, KO_SAMPLE_RATE, subtype="PCM_16")
    subprocess.run(["aplay", "-D", APLAY_DEVICE, tmp_wav])
    os.unlink(tmp_wav)

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_while_held(current_direction):
    """Record audio while button is held. Returns (audio, new_direction).
    new_direction is None if no switch change occurred."""
    q         = queue.Queue()
    recording = True

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if recording:
            q.put(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=INPUT_DEVICE,
            callback=callback
        )
        stream.start()
    except Exception as e:
        print(f"Failed to open input stream: {e}")
        return np.array([], dtype="int16"), None

    red_led.on()
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
    return audio, new_dir

# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------
def get_direction():
    """Read the physical toggle switch."""
    return "ko_to_en" if direction_switch.value == 1 else "en_to_ko"

def direction_label(d):
    return "KO->EN" if d == "ko_to_en" else "EN->KO"

def wait_for_button_or_switch(current_direction):
    """Block until record button pressed or switch changes."""
    while not exit_requested:
        if record_button.is_active:
            return "record"
        if get_direction() != current_direction:
            return "switch"
        time.sleep(0.01)
    return "exit"

def wait_for_playback_or_switch(current_direction):
    """After translation, wait for button press to play, or switch change."""
    while not exit_requested:
        if record_button.is_active:
            return "play"
        if get_direction() != current_direction:
            return "switch"
        time.sleep(0.01)
    return "exit"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
try:
    current_direction = get_direction()
    print(f"Mode: {direction_label(current_direction)}")
    print("Hold button to record. Toggle switch to change direction.\n")

    while not exit_requested:
        red_led.off()
        green_led.off()

        print(f"[{direction_label(current_direction)}] Waiting...")
        action = wait_for_button_or_switch(current_direction)

        if action == "exit" or exit_requested:
            break

        if action == "switch":
            current_direction = get_direction()
            print(f"Switched to {direction_label(current_direction)}")
            continue

        # --- Record ---
        audio, new_dir = record_while_held(current_direction)

        if exit_requested:
            break

        if new_dir:
            current_direction = new_dir
            print(f"Switched to {direction_label(current_direction)}")
            continue

        if audio.size == 0:
            print("No audio captured.")
            continue

        # --- Transcribe ---
        print("Transcribing...")
        red_led.on()
        lang = "ko" if current_direction == "ko_to_en" else "en"
        text = transcribe(audio, lang)
        red_led.off()
        print(f"Recognized : {text}")

        if not text.strip():
            print("No speech detected.")
            continue

        # --- Punctuate ---
        text = punct_model.restore_punctuation(text)
        print(f"Punctuated : {text}")

        # --- Translate ---
        print("Translating...")
        red_led.on()
        if current_direction == "ko_to_en":
            translation = translate(text, "kor_Hang", "eng_Latn")
        else:
            translation = translate(text, "eng_Latn", "kor_Hang")
        red_led.off()
        print(f"Translation: {translation}")

        # --- Wait for playback confirmation ---
        green_led.on()
        print("Press button to play translation...")
        action = wait_for_playback_or_switch(current_direction)
        green_led.off()

        if action == "exit" or exit_requested:
            break

        if action == "switch":
            current_direction = get_direction()
            print(f"Switched to {direction_label(current_direction)}")
            continue

        # --- Speak ---
        print("Speaking...")
        if current_direction == "ko_to_en":
            speak_english(translation)
        else:
            speak_korean(translation)

except KeyboardInterrupt:
    print("\nKeyboard interrupt received.")

finally:
    print("Shutting down...")
    red_led.off()
    green_led.off()
    os.system("sudo shutdown -h now")
