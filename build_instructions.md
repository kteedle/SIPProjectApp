# Build Instructions

## Running from Source

1. Create and activate virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\Activate.ps1  # Windows PowerShell
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app/main.py
```

## Building Executable
Using PyInstaller to create a standalone executable:
```bash
pyinstaller --onedir --name "ImageEdgeProcessor" --add-data "app:app" app/main.py
```

### Platform notes:

Windows: May need to include tkinter DLLs if not bundled automatically

Linux: Ensure tkinter development packages are installed

macOS: May need to specify windowed mode with --windowed

The executable will be created in the dist/ folder.