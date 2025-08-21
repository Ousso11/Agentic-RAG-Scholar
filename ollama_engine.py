import os
import platform
import subprocess
import shutil
import requests
import re
import time

def is_ollama_installed():
    return shutil.which("ollama") is not None

def install_ollama():
    system = platform.system().lower()
    if system == "darwin":
        print("Installing Ollama for macOS via Homebrew...")
        subprocess.run(["brew", "install", "ollama"], check=True)
    elif system == "linux":
        print("Installing Ollama for Linux via curl script...")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

def is_model_pulled(model_name):
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
        lines = output.splitlines()

        model_names = []
        for line in lines:
            match = re.match(r'^([^\s]+)\s+', line)
            if match and not line.startswith("NAME"):
                model_names.append(match.group(1))
        return model_name in model_names
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def pull_model(model_name):
    print(f"Pulling model: {model_name} ...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull model '{model_name}': {e}")
        raise SystemExit(1)

def is_ollama_server_running():
    try:
        response = requests.get("http://localhost:11434")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def start_ollama_server():
    if not is_ollama_server_running():
        print("Starting Ollama server...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Give the server a moment to start
            time.sleep(5) 
            if is_ollama_server_running():
                print("✅ Ollama server started successfully.")
            else:
                raise RuntimeError("Ollama server failed to start.")
        except FileNotFoundError:
            raise RuntimeError("Ollama executable not found. Please ensure it's in your PATH.")
    else:
        print("✅ Ollama server is already running.")

def add_model(model_name: str):
    if not is_model_pulled(model_name):
        print(f"Model '{model_name}' not found. Pulling...")
        pull_model(model_name)
    else:
        print(f"✅ Model '{model_name}' is already pulled.")

def ollama_setup(model_name: str):
    """
    Sets up the Ollama environment by ensuring it's installed,
    the server is running, and the specified model is pulled.
    """
    system = platform.system().lower()
    if system not in ["darwin", "linux"]:
        raise SystemExit("Only macOS and Linux are supported for automatic installation.")

    if not is_ollama_installed():
        print("Ollama not found. Installing...")
        install_ollama()
    else:
        print("✅ Ollama is already installed.")

    start_ollama_server()

    add_model(model_name)