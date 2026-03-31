import requests
import speech_recognition as sr
import threading
from mtranslate import translate
from colorama import Fore, init
from indic_transliteration import sanscript
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import os
import pyttsx3
import platform
import sys
import re
import time
from datetime import datetime
import pytz
import lmstudio as lms

# ---------------------- LLM Setup ----------------------
LLM_MODEL_NAME = "llama-3.2-3b-instruct"
try:
    model = lms.llm(LLM_MODEL_NAME)
    print(f"LLM model '{LLM_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(Fore.RED + f"Failed to load LLM model: {e}")
    model = None

# ---------------------- Global Variables ----------------------
init(autoreset=True)

speech_lock = threading.Lock()
stop_news = threading.Event()
news_thread = None

# Speech interruption
stop_speech = threading.Event()
speech_thread = None
current_engine = None
engine_lock = threading.Lock()
is_speaking = False
speaking_lock = threading.Lock()
speech_start_time = 0  # timestamp when current speech started

# ---------------------- Helper Functions ----------------------
def clean_text_for_speech(text):
    """Remove or replace characters that cause mispronunciations."""
    text = text.replace('*', '')
    return text

def speak(text, done_event=None):
    """
    Launch a separate thread to speak the text, making it interruptible.
    If done_event is provided, it will be set when speech finishes.
    """
    global speech_thread, current_engine, is_speaking, speech_start_time

    # If a speech thread is already running, signal it to stop
    if speech_thread and speech_thread.is_alive():
        stop_speech.set()
        with engine_lock:
            if current_engine:
                try:
                    current_engine.stop()
                except Exception:
                    pass
        time.sleep(0.2)

    # Clear the stop flag before starting new speech
    stop_speech.clear()

    # Clean the text for speech
    text = clean_text_for_speech(text)

    def speak_worker():
        global current_engine, is_speaking, speech_start_time
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1)

            if platform.system() == "Linux":
                try:
                    engine.setProperty('driver', 'espeak')
                except Exception:
                    pass

            with engine_lock:
                current_engine = engine

            with speaking_lock:
                is_speaking = True
                speech_start_time = time.time()  # record start time

            # Speak the entire text as one block (no sentence splitting)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            if "run loop already started" not in str(e):
                print(Fore.RED + f"Error in speech synthesis: {e}", flush=True)
        finally:
            with engine_lock:
                current_engine = None
            with speaking_lock:
                is_speaking = False
            if done_event:
                done_event.set()

    speech_thread = threading.Thread(target=speak_worker)
    speech_thread.daemon = True
    speech_thread.start()

def recognize_with_timeout(recognizer, audio, language, timeout=5):
    """Recognize speech with a timeout to prevent hanging."""
    result = [None]
    exception = [None]

    def recognize_thread():
        try:
            result[0] = recognizer.recognize_google(audio, language=language).lower()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=recognize_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None
    elif exception[0]:
        raise exception[0]
    else:
        return result[0]

def Translate_bengali_to_english(text):
    try:
        bengali_script = sanscript.transliterate(text, sanscript.ITRANS, sanscript.BENGALI)
        return translate(bengali_script, "en", "bn")
    except:
        return text

def Translate_hindi_to_english(text):
    try:
        hindi_script = sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return translate(hindi_script, "en", "hi")
    except:
        return text

def capture_speech(prompt_text=None):
    """
    Capture a single user speech utterance and return it as English text.
    If prompt_text is provided, it will be spoken before listening.
    Returns None if nothing recognized or error.
    """
    if prompt_text:
        speak(prompt_text)
        # Small delay to let the TTS settle before listening
        time.sleep(0.3)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_adjustment_damping = 0.015
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.5
    recognizer.operation_timeout = None
    recognizer.non_speaking_duration = 0.2

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 2048
    SILENCE_THRESHOLD = 1.0

    try:
        print(Fore.GREEN + "Listening...", end="", flush=True)
        audio_chunks = []
        silence_duration = 0.0
        is_recording = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE) as stream:
            print(Fore.GREEN + " (recording until 1 sec pause)", end="", flush=True)
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_chunks.append(audio_chunk)
                chunk_energy = np.sqrt(np.mean(audio_chunk ** 2))

                if chunk_energy > 0.02:
                    is_recording = True
                    silence_duration = 0.0
                elif is_recording:
                    silence_duration += CHUNK_SIZE / SAMPLE_RATE
                    if silence_duration >= SILENCE_THRESHOLD:
                        break

        audio_data = np.concatenate(audio_chunks, axis=0)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        wavfile.write(temp_wav, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))

        print("\r" + Fore.LIGHTCYAN_EX + "Recog....", end="", flush=True)

        with sr.AudioFile(temp_wav) as source:
            audio = recognizer.record(source)

        recognizer_text = ""
        is_bengali = False

        try:
            recognizer_text = recognize_with_timeout(recognizer, audio, "bn-IN", timeout=5)
            if recognizer_text is None:
                raise sr.WaitTimeoutError("Recognition timeout")
            is_bengali = True
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            try:
                recognizer_text = recognize_with_timeout(recognizer, audio, "en-US", timeout=5)
                if recognizer_text is None:
                    raise sr.WaitTimeoutError("Recognition timeout")
                is_bengali = False
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                recognizer_text = ""

        if recognizer_text:
            eng_text = Translate_bengali_to_english(recognizer_text) if is_bengali else recognizer_text
            print("\r" + Fore.BLUE + "You said" + (" (Bengali): " if is_bengali else ": ") + Fore.CYAN + eng_text)
            return eng_text
        else:
            print("\r" + Fore.YELLOW + "No text recognized", end="", flush=True)
            return None

    except Exception as e:
        print("\r" + Fore.RED + f"Error capturing speech: {e}")
        return None
    finally:
        if 'temp_wav' in locals() and os.path.exists(temp_wav):
            os.remove(temp_wav)

def news():
    """Fetch and speak top news headlines, checking for stop signal."""
    global news_thread
    try:
        # Speak the initial "Please wait" message and wait for it to finish (or be interrupted)
        print("Please wait, fetching the latest news")
        done_event = threading.Event()
        speak("Please wait, fetching the latest news", done_event=done_event)
        # Wait for the speech to finish, but allow pause interruption
        while not stop_news.is_set():
            if done_event.wait(timeout=0.2):
                break
        if stop_news.is_set():
            print("\nNews cancelled before fetching.")
            return

        # Now fetch news
        print("Fetching news from API...")
        main_url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=a9a513a6ed2b46308fcd2ee644c72be0'
        main_page = requests.get(main_url, timeout=10).json()
        articles = main_page['articles']
        head = [ar['title'] for ar in articles]
        day = ["first", "second", "third", "fourth", "fifth"]
        print(f"Got {len(head)} news articles")
        for i in range(min(len(day), len(head))):
            if stop_news.is_set():
                print("\nNews reading interrupted.")
                break
            news_text = f"Today's {day[i]} news is: {head[i]}"
            print(f"Speaking: {news_text[:60]}...")
            done_event = threading.Event()
            speak(news_text, done_event=done_event)
            # Wait for this headline to finish, but allow pause interruption
            while not stop_news.is_set():
                if done_event.wait(timeout=0.2):
                    break
            if stop_news.is_set():
                break
        print("News fetching completed")
    except Exception as e:
        error_msg = f"Sorry, I couldn't fetch the news. Error: {str(e)}"
        print(Fore.RED + error_msg, flush=True)
        speak(error_msg)
    finally:
        stop_news.clear()
        news_thread = None

def ask_llm(question):
    """Send question to LLM and stream the response to console."""
    if model is None:
        error_msg = "Sorry, the language model is not available."
        print(Fore.RED + error_msg)
        speak(error_msg)
        return

    concise_prompt = "Keep your answer short and to the point. Avoid using asterisks or markdown. "
    full_prompt = concise_prompt + question

    try:
        print(Fore.CYAN + "\n[LLM] Thinking...")
        full_response = ""

        try:
            stream = model.respond(full_prompt, stream=True)
            if hasattr(stream, '__iter__') and not isinstance(stream, str):
                print(Fore.GREEN + "[LLM] " + Fore.WHITE, end="", flush=True)
                for chunk in stream:
                    if hasattr(chunk, 'content'):
                        text = chunk.content
                    elif isinstance(chunk, dict):
                        text = chunk.get('content', '')
                    else:
                        text = str(chunk)
                    print(text, end="", flush=True)
                    full_response += text
                print()
            else:
                full_response = _extract_text(stream)
                print(Fore.GREEN + "[LLM] " + full_response)
        except TypeError:
            result = model.respond(full_prompt)
            full_response = _extract_text(result)
            print(Fore.GREEN + "[LLM] " + full_response)

        if full_response:
            speak(full_response)
        else:
            speak("I'm sorry, I couldn't generate a response.")

    except Exception as e:
        error_msg = f"Error getting response from LLM: {e}"
        print(Fore.RED + error_msg)
        speak("Sorry, I encountered an error while processing your question.")

def _extract_text(obj):
    if hasattr(obj, 'content'):
        return obj.content
    elif hasattr(obj, 'text'):
        return obj.text
    else:
        return str(obj)

def Speech_to_text_Python():
    global news_thread, stop_news, stop_speech, is_speaking, speech_start_time
    while True:
        eng_text = capture_speech()

        if eng_text is None:
            continue

        # If the assistant is speaking, only allow "pause" command
        with speaking_lock:
            speaking = is_speaking
        if speaking:
            # Check for pause command
            if re.search(r'\bpause\b', eng_text, re.IGNORECASE):
                # If speech started less than 2 seconds ago, ignore (likely TTS)
                if time.time() - speech_start_time < 2.0:
                    print(Fore.YELLOW + "Ignoring pause command (too early, likely TTS)")
                    continue
                # Otherwise, process pause
            else:
                print(Fore.YELLOW + "Assistant is speaking, please wait...")
                continue

        # ---------- COMMAND HANDLING ----------
        if re.search(r'\b(exit|terminate)\s+(?:the\s+)?(program|programme)\b', eng_text, re.IGNORECASE):
            print("\r" + Fore.RED + "Exiting program...")
            speak("Goodbye!")
            if news_thread is not None and news_thread.is_alive():
                stop_news.set()
            time.sleep(1)
            break

        elif re.search(r'\bnews\b', eng_text, re.IGNORECASE):
            if news_thread is not None and news_thread.is_alive():
                print(Fore.YELLOW + "News already playing. Say 'pause' to stop.")
            else:
                stop_news.clear()
                news_thread = threading.Thread(target=news)
                news_thread.daemon = True
                news_thread.start()

        elif re.search(r'\bpause\b', eng_text, re.IGNORECASE):
            if news_thread is not None and news_thread.is_alive():
                print("Pausing news...")
                stop_news.set()
                # Stop any ongoing speech
                with engine_lock:
                    if current_engine:
                        try:
                            current_engine.stop()
                        except Exception:
                            pass
                speak("Pausing news.")
            elif speech_thread and speech_thread.is_alive():
                print("Pausing speech...")
                stop_speech.set()
                with engine_lock:
                    if current_engine:
                        try:
                            current_engine.stop()
                        except Exception:
                            pass
            else:
                print("Nothing to pause.")

        elif re.search(r'\b(where (?:are we|am i)|(?:our|my) location)\b', eng_text, re.IGNORECASE):
            try:
                ipAdd = requests.get('https://api.ipify.org').text
                url = f'https://get.geojs.io/v1/ip/geo/{ipAdd}.json'
                geo_requests = requests.get(url)
                geo_data = geo_requests.json()
                city = geo_data.get('city', 'unknown city')
                country = geo_data.get('country', 'unknown country')
                location_msg = f"We are in {city} city of {country}."
                print(Fore.CYAN + location_msg)
                speak(location_msg)
            except Exception as e:
                print(Fore.RED + f"Location error: {e}")
                speak("Sorry sir, due to network issue I am not able to find where we are.")

        elif re.search(r'\b(?:what is the time|what is today\'s date|what is todays date|what is today date|what is the day today|what day is today)\b', eng_text, re.IGNORECASE):
            try:
                ipAdd = requests.get('https://api.ipify.org').text
                url = f'https://get.geojs.io/v1/ip/geo/{ipAdd}.json'
                geo_requests = requests.get(url)
                geo_data = geo_requests.json()
                timezone = geo_data.get("timezone")
                tz = pytz.timezone(timezone)
                now = datetime.now(tz)
                if re.search(r'\bwhat is the time\b', eng_text, re.IGNORECASE):
                    time_msg = f"The current time is {now.strftime('%I:%M %p')}."
                    print(Fore.CYAN + time_msg)
                    speak(time_msg)
                elif re.search(r'\bwhat is today\'s date\b', eng_text, re.IGNORECASE) or re.search(r'\bwhat is todays date\b', eng_text, re.IGNORECASE) or re.search(r'\bwhat is today date\b', eng_text, re.IGNORECASE):
                    date_msg = f"Today's date is {now.strftime('%d, %m, %Y')}."
                    print(Fore.CYAN + date_msg)
                    speak(date_msg)
                elif re.search(r'\bwhat is the day\b', eng_text, re.IGNORECASE) or re.search(r'\bwhat day is today\b', eng_text, re.IGNORECASE):
                    day_msg = f"Today is {now.strftime('%A')}."
                    print(Fore.CYAN + day_msg)
                    speak(day_msg)
            except Exception as e:
                print(Fore.RED + f"Location error: {e}")
                speak("Sorry sir, due to network issue I am not able to find where we are.")

        elif re.search(r'\b(goodbye|good bye|bye)\b', eng_text, re.IGNORECASE):
            farewell = "Goodbye! Have a nice day!"
            print("\r" + Fore.MAGENTA + farewell)
            speak(farewell)

        elif re.search(r'\b(hi|hello)\b', eng_text, re.IGNORECASE):
            greeting = "Hello! My name is ARIA. I have been made by Angshuk Dutta. How may I help you?"
            print("\r" + Fore.MAGENTA + greeting)
            speak(greeting)

        elif re.search(r'\bhow are you\b', eng_text, re.IGNORECASE):
            how_response = "I am good, thank you. How are you?"
            print("\r" + Fore.MAGENTA + how_response)
            speak(how_response)

        elif re.search(r'\b(i have a (question|doubt|query)|answer me|solve my (question|query|doubt)|answer\s+my\s*(problem|question)|solve\s+my\s*(problem|question|doubt)|mr\.?\s*ansar)\b', eng_text, re.IGNORECASE):
            question = capture_speech(prompt_text="Please go ahead with your question.")
            if question:
                ask_llm(question)
            else:
                no_input_msg = "I didn't hear your question. Please try again."
                print(Fore.YELLOW + no_input_msg)
                speak(no_input_msg)

        else:
            pass

# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    speech_thread = threading.Thread(target=Speech_to_text_Python)
    speech_thread.daemon = True
    speech_thread.start()
    speech_thread.join()