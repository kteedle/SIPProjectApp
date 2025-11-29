"""
Main entry point for the Image Edge Processing application.
"""
import tkinter as tk
from app.gui import ImageProcessingApp

def main():
    """Launch the main application window."""
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()



