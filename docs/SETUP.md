# Setup: Colab, Mac and Windows

**In class, everyone uses Google Colab** with its free GPU: nothing to install on your laptop, and the
in-class time budgets were measured there. The Mac and Windows setups are optional, for anyone who wants to run
the code locally after the lecture. A laptop runs everything on the CPU (or the Apple GPU on Apple-silicon
Macs), so training is slower than on Colab: on an Apple-silicon Mac about half as many training passes fit in
the same time, and on a CPU-only laptop far fewer. The notebooks say how to use smaller runs or the pre-trained
weights on a laptop.

Everything here is free. Nothing asks for a token, key or paid account.

| | Colab | Mac | Windows |
|---|---|---|---|
| What you need | A Google account and a browser | macOS, Python 3.10 or newer | Windows 10 or 11, Python 3.10 or newer |
| Time to set up | about 1 minute | 5 to 15 minutes the first time | 5 to 15 minutes the first time |
| Disk space | none on your laptop | about 2 GB for `.venv` plus about 0.6 GB for the roof packs | the same |
| Admin rights | no | no | no |
| Tested on | Colab with a T4 GPU | An Apple-silicon Mac (Python 3.15, PyTorch 2.15) | **Not yet on a real Windows PC** (see 3.0) |

---

## 1. Google Colab (used in class)

1. Open the lecture notebook from the link your instructor shares, and sign in to Google if asked.
2. In the menu bar, click **Runtime → Change runtime type**.
3. Under **Hardware accelerator**, choose **T4 GPU**, then click **Save**.
4. Run the first code cell (click it, then press **Shift + Enter**). It installs the course package with:

   ```
   %pip install -q "genroofs @ git+https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery.git"
   ```

   Colab already has PyTorch, so this takes a few seconds and leaves Colab's own PyTorch alone.
5. Optional: run the self-check in a new cell. The last line should read `ALL GOOD`.

   ```
   !python -m genroofs.check
   ```

The roof packs download into `/content/genroofs_data` the first time a notebook loads them (about 25 seconds),
and they are checked against their SHA-256 before use. A Colab session forgets them when it ends; the next
session downloads them again.

---

## 2. Mac

### 2.1 Check that Python 3.10 or newer is installed

1. Open **Terminal**: press **Cmd + Space**, type `Terminal`, press **Enter**.
2. Type this and press **Enter**:

   ```
   python3 --version
   ```

3. Read the answer:
   - `Python 3.10.x` or a higher number (3.11, 3.12, 3.13, ...): good, go to 2.2.
   - `Python 3.9.x` or lower, `command not found`, or a window asking to install *command line developer tools*:
     close that window and install Python as described next.

**Installing Python on a Mac** (only if needed)

1. In a browser, go to <https://www.python.org/downloads/macos/>.
2. Under **Stable Releases**, click the newest **Python 3** release, then **macOS 64-bit universal2 installer**.
3. Open the downloaded `.pkg` file and click **Continue**, **Continue**, **Agree**, **Install**. macOS asks for
   an administrator's name and password to install Python itself. If you can't give one, use Google Colab
   (section 1) instead; the lab's own setup below never needs administrator rights.
4. When it finishes, close Terminal completely (**Cmd + Q**) and open it again, then repeat step 2 above.

### 2.2 Get the project folder

Either way works:

- **Download ZIP:** open <https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery>, click the green
  **Code** button, click **Download ZIP**, then double-click the ZIP in **Downloads** to unpack it.
- **Git:** in Terminal, run the following two lines; Git places the project in a new folder named
  `generative-models-aerial-imagery` inside the folder Terminal is in:

  ```
  cd ~
  git clone https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery.git
  ```

Put the folder somewhere that is **not synced** to iCloud Drive (for example your home folder or a `Projects`
folder). Syncing the thousands of files in `.venv` makes iCloud slow and can break the setup.

### 2.3 Run the setup script

1. In Terminal, type `cd ` (with a space after it), drag the project folder from Finder into the Terminal
   window, and press **Enter**. Terminal is now inside the project folder.
2. Run:

   ```
   bash scripts/setup_mac.sh
   ```

The script prints each step as it goes:

1. It finds your newest Python 3.10+.
2. It creates `.venv` inside the project folder.
3. It upgrades `pip` inside `.venv`.
4. It installs the packages, including PyTorch and JupyterLab, into `.venv`.
5. It installs the course package and runs the self-check.

It never uses `sudo` and installs nothing outside the project folder.

### 2.4 What success looks like

The script ends with the self-check. Your version numbers will differ; what matters is the last line:

```
genroofs setup check
--------------------
Python      3.13.x
torch       2.x.x
numpy       2.x.x
Pillow      1x.x.x
matplotlib  3.x.x
safetensors 0.x.x
certifi     20xx.xx.xx
pandas      not installed (optional, for to_pandas())
gradio      not installed (optional, for the web app)
genroofs    0.1.0.dev0
Device      mps (Apple GPU (mps))
Cache       ~/Projects/generative-models-aerial-imagery/data/cache (xxx.x GB free)
Release     data-v1 answers
Training    2 steps of a tiny autoencoder on mps in 0.x s
--------------------
ALL GOOD
```

On an Apple-silicon Mac the device is `mps` (the Apple GPU); on an Intel Mac it is `cpu`. If the last line says
**Fix these**, follow the numbered list it prints, then run the script again.

### 2.5 Using it later

Open Terminal, go into the project folder as in 2.3 step 1, then:

```
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. From there:

| To do this | Run |
|---|---|
| Open the notebooks in JupyterLab | `jupyter lab` |
| Run the course package's quick tests (about 30 seconds) | `python -m pytest -q tests -k genroofs` |
| Run every test, including the pack builder's (a few minutes) | `python -m pytest -q tests` |
| Run the self-check again | `python -m genroofs.check` |
| Add the optional web page and pandas | `python -m pip install gradio pandas` |
| Leave the environment | `deactivate` |

**What a good test run looks like.** The quick tests end with a line like
`157 passed, 2 skipped, 6 warnings in 23.36s`, and all tests with `207 passed, 2 skipped, 6 warnings`
(the counts grow as tests are added). Both are expected:

- **2 skipped:** the tests for the optional extras, pandas and the Gradio web page, are skipped when those
  extras are not installed. After `python -m pip install gradio pandas` they run too.
- **6 warnings:** each says `FigureCanvasAgg is non-interactive, and thus cannot be shown`. The tests draw
  pictures off-screen, where there is no window to show them in. They are harmless.

Anything listed as `failed` or `error` is a real problem: send the full test output to the instructor.

---

## 3. Windows

### 3.0 Status: provided, not yet tested on a real Windows PC

In class, Windows users follow the lecture on **Google Colab** (section 1), like everyone else. The Windows
setup below is included for anyone who wants a local copy afterwards. Its current state:

| Checked | Not yet checked |
|---|---|
| The script `scripts\setup_windows.ps1` parses with no errors in PowerShell 7.4 | A run on a real Windows 10 or 11 PC |
| Its steps run end to end under PowerShell 7.4 on Linux (with Linux paths swapped in): find Python, create `.venv`, upgrade pip, install, run the self-check | Windows PowerShell 5.1, the version built into Windows |
| It uses plain ASCII only, which Windows PowerShell 5.1 needs | The `py` launcher, the Windows build of PyTorch, and Windows file-path length limits |
| The Python package, its tests and the self-check pass on macOS and Linux | The tests and the self-check on Windows |

If you try it and anything fails, send the instructor the last 40 lines of the script's output, or the full
output of `.venv\Scripts\python.exe -m genroofs.check`. The self-check never prints your user name; pip's own
lines in the script's output can include folder paths, so look them over before sharing. The steps below are
complete and match the Mac setup step for step.

### 3.1 Check that Python 3.10 or newer is installed

1. Open **PowerShell**: click **Start**, type `PowerShell`, click **Windows PowerShell**.
2. Type this and press **Enter**:

   ```
   py --version
   ```

3. Read the answer:
   - `Python 3.10.x` or a higher number: good, go to 3.2.
   - An error such as *'py' is not recognized*, or a version below 3.10: install Python as described next.

**Installing Python on Windows** (only if needed)

1. In a browser, go to <https://www.python.org/downloads/windows/>.
2. Under **Stable Releases**, click the newest **Python 3** release's **Windows installer (64-bit)**.
3. Open the downloaded `.exe`. On the first screen:
   - Tick **Add python.exe to PATH** at the bottom.
   - If you have no administrator rights, untick **Use admin privileges when installing py.exe**.
   - Click **Install Now**.
4. If the last screen offers **Disable path length limit** and you have administrator rights, click it (it helps
   with long folder names); otherwise skip it. Click **Close**.
5. Close every PowerShell window, open a new one, and repeat step 2 above.

### 3.2 Get the project folder

- **Download ZIP:** open <https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery>, click the green
  **Code** button, click **Download ZIP**. In **Downloads**, right-click the ZIP, choose **Extract All...**, and
  extract it to a **short folder that OneDrive does not sync**, for example `C:\lab`.
- **Git:** in PowerShell, run these two lines:

  ```
  cd C:\
  git clone https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery.git lab
  ```

A short path matters on Windows: PyTorch has long file names, and very long folder paths can make the install
fail (see Troubleshooting). OneDrive syncing the thousands of files in `.venv` is slow and can break the setup.

### 3.3 Run the setup script

1. In **File Explorer**, open the project folder (the one that contains `scripts` and `src`).
2. Click the address bar at the top, type `powershell`, and press **Enter**. A PowerShell window opens inside
   that folder.
3. Run:

   ```
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```

`-ExecutionPolicy Bypass` lets this one script run this one time; it changes no Windows setting. The script
does the same five steps as the Mac script, using the `py` launcher, and needs no administrator rights.

### 3.4 What success looks like

The same self-check as on a Mac, ending in `ALL GOOD`. On most Windows laptops the device line reads
`Device      cpu (CPU, N threads)`; that is expected.

### 3.5 Using it later

Open PowerShell in the project folder (as in 3.3 steps 1 and 2), then run the program inside `.venv` directly.
This needs no activation step and no change to Windows' script policy:

| To do this | Run |
|---|---|
| Open the notebooks in JupyterLab | `.venv\Scripts\python.exe -m jupyter lab` |
| Run the course package's quick tests (about 30 seconds) | `.venv\Scripts\python.exe -m pytest -q tests -k genroofs` |
| Run every test, including the pack builder's (a few minutes) | `.venv\Scripts\python.exe -m pytest -q tests` |
| Run the self-check again | `.venv\Scripts\python.exe -m genroofs.check` |
| Add the optional web page and pandas | `.venv\Scripts\python.exe -m pip install gradio pandas` |

The test results should look like the Mac's in 2.5: `passed`, `2 skipped` and `6 warnings` are expected;
`failed` or `error` is not.

---

## 4. What the self-check checks

| Line | Meaning | If it is wrong |
|---|---|---|
| `Python` | The Python inside `.venv` | Below 3.10: install a newer Python, delete `.venv`, run the setup script again |
| `torch` ... `certifi` | The packages the lab needs | `MISSING`: run the setup script again |
| `pandas`, `gradio` | Optional extras | Nothing to fix; install them only if you want them |
| `Device` | Where training runs: `cuda` (NVIDIA GPU), `mps` (Apple GPU) or `cpu` | Nothing to fix; `cpu` is just slower |
| `Cache` | Where the roof packs are kept, and the free disk space there | Under 1 GB free: free some space, or set `GENROOFS_DATA` (below) |
| `Release` | Whether the roof packs on GitHub can be reached | `did NOT answer`: check the internet connection (see Troubleshooting) |
| `Training` | Two training steps of a tiny autoencoder | An error here: restart, run the check again, and send the output to the instructor |

The check prints `~` instead of your home folder, so its output never shows your user name and is safe to share.

---

## 5. Where things are stored

| What | Where | Size |
|---|---|---|
| Python packages | `.venv` inside the project folder | about 2 GB |
| Roof packs and their decoded copies | `data/cache` inside the project folder (on Colab: `/content/genroofs_data`) | about 0.6 GB once both packs are loaded |
| pip's download cache | your user cache folder (pip's normal place) | varies |

To keep the packs somewhere else, set `GENROOFS_DATA` before starting Jupyter. On a Mac (in Terminal):

```
export GENROOFS_DATA="$HOME/genroofs_data"
```

On Windows (in PowerShell, for the current window):

```
$env:GENROOFS_DATA = "D:\genroofs_data"
```

**To remove everything:** delete the `.venv` folder and the `data` folder inside the project folder, or delete
the whole project folder. Nothing was installed anywhere else.

---

## 6. Troubleshooting

| What you see | What to do |
|---|---|
| `Python 3.10 or newer was not found` | Install Python as in 2.1 or 3.1, open a new Terminal or PowerShell window, run the script again |
| Mac: a window asks to install *command line developer tools* | Close it and install Python from python.org (2.1); the lab doesn't need the developer tools |
| `... exists but has no Python 3.10 or newer` | An old `.venv` is in the folder. Delete the `.venv` folder (in Finder or File Explorer), run the script again |
| Mac, Intel processor: PyTorch fails to install | PyTorch no longer publishes new builds for Intel Macs. Use Google Colab (section 1) |
| Windows: `running scripts is disabled on this system` | Type the full command from 3.3, including `-ExecutionPolicy Bypass` |
| Windows: an install error mentioning a long path or `No such file or directory` | Move the project folder to a short path such as `C:\lab`, delete `.venv`, run the script again |
| `SSL: CERTIFICATE_VERIFY_FAILED`, or `Release ... did NOT answer` on a work or campus network | The network inspects secure connections. Ask IT for its certificate bundle file and point `SSL_CERT_FILE` at it (as with `GENROOFS_DATA` above), or use another network or Colab |
| pip stops with a timeout or connection error | Run the script again; it reuses what is already installed and continues |
| The check says only a little disk space is free | Free some space, or set `GENROOFS_DATA` to a drive with more room |
| Anything else | Run the self-check, and send its full output (it contains no user name) to the instructor |
