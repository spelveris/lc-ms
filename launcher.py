"""Launcher script for LC-MS Analysis app."""

import sys
import os
import time
import threading

class LoadingSpinner:
    """Display a colorful spinning animation with timer during loading."""

    # ANSI color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self):
        self.running = False
        self.start_time = None
        self.thread = None
        self.message = "Loading"
        # Colorful spinning animation
        self.frames = [
            f"{self.CYAN}⣾{self.RESET}",
            f"{self.CYAN}⣽{self.RESET}",
            f"{self.BLUE}⣻{self.RESET}",
            f"{self.BLUE}⢿{self.RESET}",
            f"{self.MAGENTA}⡿{self.RESET}",
            f"{self.MAGENTA}⣟{self.RESET}",
            f"{self.CYAN}⣯{self.RESET}",
            f"{self.CYAN}⣷{self.RESET}",
        ]

    def _update(self):
        frame_idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            spinner = self.frames[frame_idx % len(self.frames)]
            timer = f"{self.YELLOW}{elapsed:.1f}s{self.RESET}"
            print(f"\r  {spinner} {self.message}  {timer}", end="    ")
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.1)

    def start(self, message="Loading"):
        self.message = message
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        elapsed = time.time() - self.start_time
        check = f"{self.GREEN}✓{self.RESET}"
        timer = f"{self.GREEN}{elapsed:.1f}s{self.RESET}"
        print(f"\r  {check} {self.message}  {timer}    ")

# Required for PyInstaller multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Set up paths for frozen executable
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app', 'main.py')
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app', 'main.py')

    # Change to app directory
    os.chdir(os.path.dirname(app_path))

    # Colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    print()
    print(f"  {BOLD}{CYAN}LC-MS Analysis{RESET}")
    print(f"  {DIM}v1.0{RESET}")
    print()

    # Create loading HTML page that shows immediately and redirects when ready
    import webbrowser
    import tempfile

    LOADING_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>LC-MS Analysis - Loading</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .spinner {
            width: 60px;
            height: 60px;
            position: relative;
            animation: spinner-rotate 1.2s linear infinite;
        }
        .dot {
            position: absolute;
            width: 10px;
            height: 10px;
            background: #215CAF;
            border-radius: 50%;
            animation: spinner-fade 1.2s linear infinite;
        }
        .dot:nth-child(1) { top: 0; left: 25px; animation-delay: 0s; }
        .dot:nth-child(2) { top: 7px; left: 43px; animation-delay: -0.15s; }
        .dot:nth-child(3) { top: 25px; left: 50px; animation-delay: -0.3s; }
        .dot:nth-child(4) { top: 43px; left: 43px; animation-delay: -0.45s; }
        .dot:nth-child(5) { top: 50px; left: 25px; animation-delay: -0.6s; }
        .dot:nth-child(6) { top: 43px; left: 7px; animation-delay: -0.75s; }
        .dot:nth-child(7) { top: 25px; left: 0; animation-delay: -0.9s; }
        .dot:nth-child(8) { top: 7px; left: 7px; animation-delay: -1.05s; }
        @keyframes spinner-rotate { 100% { transform: rotate(360deg); } }
        @keyframes spinner-fade {
            0%, 100% { opacity: 0.2; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
        }
        .text { margin-top: 24px; font-size: 16px; color: #333; }
        .timer { margin-top: 8px; font-size: 12px; color: #888; }
    </style>
</head>
<body>
    <div class="spinner">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div><div class="dot"></div>
        <div class="dot"></div><div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
    <div class="text">Please wait...</div>
    <div class="timer" id="timer">0.0s</div>
    <script>
        const start = Date.now();
        const target = 'http://localhost:8501';
        const timerEl = document.getElementById('timer');

        function checkServer() {
            const elapsed = ((Date.now() - start) / 1000).toFixed(1);
            timerEl.textContent = elapsed + 's';

            fetch(target, { mode: 'no-cors' })
                .then(() => { window.location.href = target; })
                .catch(() => { setTimeout(checkServer, 500); });
        }
        checkServer();
    </script>
</body>
</html>"""

    # Write loading page to temp file and open immediately
    loading_file = os.path.join(tempfile.gettempdir(), 'lcms_loading.html')
    with open(loading_file, 'w') as f:
        f.write(LOADING_HTML)

    # Open loading page immediately in browser
    webbrowser.open('file://' + loading_file)

    # Terminal spinner
    spinner = LoadingSpinner()
    spinner.start("Starting server")

    print(f"  {CYAN}http://localhost:8501{RESET}")
    print()

    # Run streamlit directly (not via subprocess)
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false"
    ]

    from streamlit.web import cli as stcli
    stcli.main()
