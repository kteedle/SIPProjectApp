"""
Tkinter GUI for the image processing pipeline.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import datetime
import time

from app.io_utils import load_image, save_image, get_image_bands
from app.processing import (
    apply_smoothing, 
    apply_sobel, 
    threshold_image,
    generate_box_kernel,
    generate_gaussian_kernel,
    process_custom_rgb_bands,
    process_all_bands,
    process_single_band
    
)
from app.metrics import compare_edge_maps

class ImageProcessingApp:
    """Main application GUI for image edge processing."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Edge Processor - Enhanced")
        self.root.geometry("1400x900")
        
        # State variables
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.image_bands: List[Dict] = []
        self.results: Dict[str, Any] = {}
        self.processing_thread: Optional[threading.Thread] = None
        self.abort_processing = False

        # Region of interest (ROI) selection state (x0, y0, x1, y1 in image coordinates)
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_canvas_start: Optional[Tuple[int, int]] = None
        self._roi_rect_id: Optional[int] = None

        # References for comparison images (for zoom)
        self.box_smoothed_image: Optional[np.ndarray] = None
        self.box_edges_image: Optional[np.ndarray] = None
        self.gaussian_smoothed_image: Optional[np.ndarray] = None
        self.gaussian_edges_image: Optional[np.ndarray] = None
        
        self.setup_logging()
        self.create_widgets()
        
    def setup_logging(self):
        """Setup logging to text widget."""
        self.log_messages: List[str] = []
        logging.basicConfig(level=logging.INFO)
        
    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_entry + "\n")
            self.log_text.see(tk.END)
        print(log_entry)
        
    def create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # Right panel - Display
        right_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Bottom panel - Log
        bottom_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        bottom_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        
        self.create_control_panel(left_frame)
        self.create_display_panel(right_frame)
        self.create_log_panel(bottom_frame)
        
    def create_control_panel(self, parent: ttk.Frame):
        """Create the control panel with all input widgets."""
        # Create a scrollable frame for controls
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Now create all controls inside scrollable_frame instead of parent
        control_parent = scrollable_frame
        
        # File selection
        file_frame = ttk.Frame(control_parent)
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", 
                command=self.load_image).grid(row=0, column=0, sticky=tk.W)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        file_frame.columnconfigure(1, weight=1)
        
        # Band selection - RESTORED MULTI-BAND OPTIONS
        self.band_frame = ttk.LabelFrame(control_parent, text="Band Processing", padding="5")
        self.band_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.band_var = tk.StringVar(value="all_bands")
        ttk.Radiobutton(self.band_frame, text="All Bands (Color)", 
                    variable=self.band_var, value="all_bands",
                    command=self.on_band_mode_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(self.band_frame, text="Single Band", 
                    variable=self.band_var, value="single",
                    command=self.on_band_mode_change).grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(self.band_frame, text="Custom RGB Bands", 
                    variable=self.band_var, value="custom_rgb",
                    command=self.on_band_mode_change).grid(row=2, column=0, sticky=tk.W)
        
        # Single band selection (initially hidden)
        self.single_band_frame = ttk.Frame(self.band_frame)
        self.single_band_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(self.single_band_frame, text="Select band:").grid(row=0, column=0, sticky=tk.W)
        
        self.single_band_var = tk.StringVar(value="0")
        self.single_band_combo = ttk.Combobox(self.single_band_frame, textvariable=self.single_band_var, 
                                            state="readonly", width=15)
        self.single_band_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.single_band_frame.grid_remove()  # Hide initially
        
        # Custom RGB band selection (initially hidden)
        self.custom_rgb_frame = ttk.Frame(self.band_frame)
        self.custom_rgb_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(self.custom_rgb_frame, text="Red Band:").grid(row=0, column=0, sticky=tk.W)
        self.red_band_var = tk.StringVar(value="0")
        self.red_band_combo = ttk.Combobox(self.custom_rgb_frame, textvariable=self.red_band_var, 
                                        state="readonly", width=12)
        self.red_band_combo.grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        
        ttk.Label(self.custom_rgb_frame, text="Green Band:").grid(row=0, column=1, sticky=tk.W)
        self.green_band_var = tk.StringVar(value="1") 
        self.green_band_combo = ttk.Combobox(self.custom_rgb_frame, textvariable=self.green_band_var,
                                            state="readonly", width=12)
        self.green_band_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 5))
        
        ttk.Label(self.custom_rgb_frame, text="Blue Band:").grid(row=0, column=2, sticky=tk.W)
        self.blue_band_var = tk.StringVar(value="2")
        self.blue_band_combo = ttk.Combobox(self.custom_rgb_frame, textvariable=self.blue_band_var,
                                        state="readonly", width=12)
        self.blue_band_combo.grid(row=1, column=2, sticky=tk.W)
        
        self.custom_rgb_frame.grid_remove()  # Hide initially
        
        # Smoothing options
        smooth_frame = ttk.LabelFrame(control_parent, text="Smoothing", padding="5")
        smooth_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.smooth_var = tk.StringVar(value="box")
        ttk.Radiobutton(smooth_frame, text="Box Filter", 
                    variable=self.smooth_var, value="box").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(smooth_frame, text="Gaussian", 
                    variable=self.smooth_var, value="gaussian").grid(row=0, column=1, sticky=tk.W)
        
        # Kernel size
        ttk.Label(smooth_frame, text="Kernel Size:").grid(row=1, column=0, sticky=tk.W)
        self.kernel_size_var = tk.StringVar(value="3")
        kernel_spin = ttk.Spinbox(smooth_frame, from_=3, to=31, increment=2, 
                                textvariable=self.kernel_size_var, width=10)
        kernel_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Sigma (for Gaussian)
        ttk.Label(smooth_frame, text="Sigma:").grid(row=2, column=0, sticky=tk.W)
        self.sigma_var = tk.StringVar(value="1.0")
        sigma_spin = ttk.Spinbox(smooth_frame, from_=0.1, to=10.0, increment=0.1,
                                textvariable=self.sigma_var, width=10)
        sigma_spin.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Sobel options
        sobel_frame = ttk.LabelFrame(control_parent, text="Sobel Operator", padding="5")
        sobel_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.sobel_var = tk.StringVar(value="both")
        # ttk.Radiobutton(sobel_frame, text="X Direction", 
        #             variable=self.sobel_var, value="x").grid(row=0, column=0, sticky=tk.W)
        # ttk.Radiobutton(sobel_frame, text="Y Direction", 
        #             variable=self.sobel_var, value="y").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(sobel_frame, text="Sobel Operator", 
                    variable=self.sobel_var, value="both").grid(row=2, column=0, sticky=tk.W)
        
        # Threshold options
        threshold_frame = ttk.LabelFrame(control_parent, text="Threshold", padding="5")
        threshold_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Threshold Value:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.2)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        self.threshold_mode = tk.StringVar(value="relative")
        threshold_mode_frame = ttk.Frame(threshold_frame)
        threshold_mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Radiobutton(threshold_mode_frame, text="Relative (0-1 of max gradient)", 
                    variable=self.threshold_mode, value="relative").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(threshold_mode_frame, text="Absolute (0-1 value)", 
                    variable=self.threshold_mode, value="absolute").grid(row=1, column=0, sticky=tk.W)
        
        # Comparison toggle
        self.compare_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(control_parent, text="Compare Box vs Gaussian", 
                    variable=self.compare_var).grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        
        # Action buttons
        button_frame = ttk.Frame(control_parent)
        button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Preview", 
                command=self.preview_processing).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Run Full Processing", 
                command=self.run_processing).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Abort", 
                command=self.abort_processing_command).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Clear", 
                command=self.clear_all).grid(row=0, column=3, padx=5)
        
        ttk.Button(button_frame, text="Save Results", 
                command=self.save_results).grid(row=1, column=0, padx=(0, 5), pady=(5, 0))
        ttk.Button(button_frame, text="Export Metrics", 
                command=self.export_metrics).grid(row=1, column=1, padx=5, pady=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_parent, mode='indeterminate')
        self.progress.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_parent, textvariable=self.status_var).grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        
        # Configure column weights for responsive layout
        control_parent.columnconfigure(0, weight=1)
        for frame in [file_frame, self.band_frame, smooth_frame, sobel_frame, threshold_frame, button_frame]:
            frame.columnconfigure(0, weight=1)

    def on_band_mode_change(self):
        """Show/hide band selection widgets based on mode."""
        if self.band_var.get() == "single":
            self.single_band_frame.grid()
            self.custom_rgb_frame.grid_remove()
            self.update_band_combos()
        elif self.band_var.get() == "custom_rgb":
            self.single_band_frame.grid_remove()
            self.custom_rgb_frame.grid()
            self.update_band_combos()
        else:
            self.single_band_frame.grid_remove()
            self.custom_rgb_frame.grid_remove()
            
    def update_band_combos(self):
        """Update all band combo boxes with available bands."""
        band_descriptions = []
        for band in self.image_bands:
            band_descriptions.append(f"{band['index']}: {band['description']}")
        
        # Update all combo boxes
        self.single_band_combo['values'] = band_descriptions
        self.red_band_combo['values'] = band_descriptions
        self.green_band_combo['values'] = band_descriptions  
        self.blue_band_combo['values'] = band_descriptions
        
        # Set default values if not already set
        if band_descriptions:
            if not self.single_band_combo.get():
                self.single_band_combo.set(band_descriptions[0])
            if not self.red_band_combo.get():
                self.red_band_combo.set(band_descriptions[0])
            if not self.green_band_combo.get() and len(band_descriptions) > 1:
                self.green_band_combo.set(band_descriptions[1])
            if not self.blue_band_combo.get() and len(band_descriptions) > 2:
                self.blue_band_combo.set(band_descriptions[2])
        
    def create_display_panel(self, parent: ttk.Frame):
        """Create the display panel for showing images."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Create tabs
        self.original_tab = ttk.Frame(self.notebook)
        self.smoothed_tab = ttk.Frame(self.notebook)
        self.gradient_tab = ttk.Frame(self.notebook)
        self.edges_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.original_tab, text="Original")
        self.notebook.add(self.smoothed_tab, text="Smoothed")
        self.notebook.add(self.gradient_tab, text="Gradient")
        self.notebook.add(self.edges_tab, text="Edges")
        self.notebook.add(self.comparison_tab, text="Comparison")
        
        # Canvas for each tab
        self.original_canvas = tk.Canvas(self.original_tab, bg='white')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.smoothed_canvas = tk.Canvas(self.smoothed_tab, bg='white')
        self.smoothed_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.gradient_canvas = tk.Canvas(self.gradient_tab, bg='white')
        self.gradient_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.edges_canvas = tk.Canvas(self.edges_tab, bg='white')
        self.edges_canvas.pack(fill=tk.BOTH, expand=True)

        # Enable ROI selection on original canvas
        self.original_canvas.bind("<ButtonPress-1>", self._on_original_mouse_down)
        self.original_canvas.bind("<B1-Motion>", self._on_original_mouse_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self._on_original_mouse_up)

        # Comparison tab: 4 images (2x2) + metrics
        comparison_images_frame = ttk.Frame(self.comparison_tab)
        comparison_images_frame.pack(fill=tk.BOTH, expand=True)
        comparison_images_frame.rowconfigure(0, weight=1)
        comparison_images_frame.rowconfigure(1, weight=1)
        comparison_images_frame.columnconfigure(0, weight=1)
        comparison_images_frame.columnconfigure(1, weight=1)

        # Box filter smoothed
        ttk.Label(comparison_images_frame, text="Box - Smoothed").grid(row=0, column=0, sticky=tk.W)
        self.box_smoothed_canvas = tk.Canvas(comparison_images_frame, bg='white')
        self.box_smoothed_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Box filter edges
        ttk.Label(comparison_images_frame, text="Box - Edges").grid(row=0, column=1, sticky=tk.W)
        self.box_edges_canvas = tk.Canvas(comparison_images_frame, bg='white')
        self.box_edges_canvas.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Gaussian smoothed
        ttk.Label(comparison_images_frame, text="Gaussian - Smoothed").grid(row=1, column=0, sticky=tk.W)
        self.gaussian_smoothed_canvas = tk.Canvas(comparison_images_frame, bg='white')
        self.gaussian_smoothed_canvas.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Gaussian edges
        ttk.Label(comparison_images_frame, text="Gaussian - Edges").grid(row=1, column=1, sticky=tk.W)
        self.gaussian_edges_canvas = tk.Canvas(comparison_images_frame, bg='white')
        self.gaussian_edges_canvas.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Bind zoom behavior (1x1 resolution) for comparison canvases (sync all four)
        self.box_smoothed_canvas.bind("<Button-1>", lambda e: self._on_comparison_zoom(self.box_smoothed_canvas, e))
        self.box_edges_canvas.bind("<Button-1>", lambda e: self._on_comparison_zoom(self.box_edges_canvas, e))
        self.gaussian_smoothed_canvas.bind("<Button-1>", lambda e: self._on_comparison_zoom(self.gaussian_smoothed_canvas, e))
        self.gaussian_edges_canvas.bind("<Button-1>", lambda e: self._on_comparison_zoom(self.gaussian_edges_canvas, e))

        # Right-click anywhere in comparison images to reset all zooms
        self.box_smoothed_canvas.bind("<Button-3>", lambda e: self._reset_comparison_zoom())
        self.box_edges_canvas.bind("<Button-3>", lambda e: self._reset_comparison_zoom())
        self.gaussian_smoothed_canvas.bind("<Button-3>", lambda e: self._reset_comparison_zoom())
        self.gaussian_edges_canvas.bind("<Button-3>", lambda e: self._reset_comparison_zoom())
        
        # Metrics display
        metrics_frame = ttk.Frame(self.comparison_tab)
        metrics_frame.pack(fill=tk.BOTH, expand=False)
        
        ttk.Label(metrics_frame, text="Comparison Metrics:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(5, 0))
        self.metrics_text = tk.Text(metrics_frame, height=12, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
    def create_log_panel(self, parent: ttk.Frame):
        """Create the logging panel."""
        self.log_text = tk.Text(parent, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("All supported", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.jp2 *.img"),
                ("Geospatial", "*.tif *.tiff *.jp2 *.img"),
                ("Regular images", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading image...")
                self.log(f"Loading image: {file_path}")
                
                # Load with max size for display
                self.original_image = load_image(file_path, max_size=(12000, 12000))
                self.current_image = self.original_image.copy()
                self.file_label.config(text=os.path.basename(file_path))
                
                # Display the image
                self.display_image(self.original_image, self.original_canvas)
                
                # Get band information
                self.image_bands = get_image_bands(file_path)
                self.log(f"Successfully loaded image: {file_path}")
                self.log(f"Image shape: {self.original_image.shape}")
                self.log(f"Available bands: {len(self.image_bands)}")
                for band in self.image_bands:
                    self.log(f"  Band {band['index']}: {band['description']}")
                
                # Update band selection based on number of bands
                self.update_band_controls()
                self.status_var.set("Ready")
                
            except Exception as e:
                error_msg = f"Failed to load image: {str(e)}"
                messagebox.showerror("Error", error_msg)
                self.log(f"Error loading image: {error_msg}")
                # Log detailed error information
                import traceback
                self.log(f"Detailed error: {traceback.format_exc()}")
                self.status_var.set("Error loading image")
                
                # Clear any partial state
                self.original_image = None
                self.current_image = None
                self.image_bands = []
                self.file_label.config(text="No file selected")
                
                # Clear the canvas and ROI
                self.original_canvas.delete("all")
                self.roi = None
                self._roi_rect_id = None

    def update_band_controls(self):
        """Update band controls based on available bands."""
        band_descriptions = []
        for band in self.image_bands:
            band_descriptions.append(f"{band['index']}: {band['description']}")
        
        # Update all combo boxes
        self.single_band_combo['values'] = band_descriptions
        self.red_band_combo['values'] = band_descriptions
        self.green_band_combo['values'] = band_descriptions  
        self.blue_band_combo['values'] = band_descriptions
        
        # Set default values
        if band_descriptions:
            if not self.single_band_combo.get():
                self.single_band_combo.set(band_descriptions[0])
            
            # For RGB, use first 3 bands by default, or available bands
            available_bands = min(3, len(band_descriptions))
            default_bands = band_descriptions[:available_bands]
            
            if not self.red_band_combo.get():
                self.red_band_combo.set(default_bands[0])
            if not self.green_band_combo.get() and len(default_bands) > 1:
                self.green_band_combo.set(default_bands[1])
            if not self.blue_band_combo.get() and len(default_bands) > 2:
                self.blue_band_combo.set(default_bands[2])
            elif len(default_bands) == 2:
                self.blue_band_combo.set(default_bands[0])  # Use red band for blue if only 2 bands
            elif len(default_bands) == 1:
                self.green_band_combo.set(default_bands[0])
                self.blue_band_combo.set(default_bands[0])
        
        # Disable "All Bands (Color)" option if image has more than 3 bands
        if len(self.image_bands) > 3:
            # Find and disable the "All Bands" radio button
            for widget in self.band_frame.winfo_children():
                if isinstance(widget, ttk.Radiobutton) and widget.cget('value') == 'all_bands':
                    widget.config(state='disabled')
                    self.log("'All Bands (Color)' disabled - image has more than 3 bands")
                    # Force selection to another option if currently selected
                    if self.band_var.get() == 'all_bands':
                        self.band_var.set('custom_rgb')
                        self.on_band_mode_change()
        else:
            # Enable "All Bands" if 3 or fewer bands
            for widget in self.band_frame.winfo_children():
                if isinstance(widget, ttk.Radiobutton) and widget.cget('value') == 'all_bands':
                    widget.config(state='normal')
                
    def display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """Display an image on a canvas. Handles both single and multi-band images."""
        if image is None:
            return
            
        try:
            # Convert numpy array to displayable format
            if image.dtype != np.uint8:
                # Handle different data types and normalize
                if np.max(image) <= 1.0 and np.min(image) >= 0:
                    # Image is in [0,1] range
                    image_display = (image * 255).astype(np.uint8)
                else:
                    # Normalize to 0-255
                    image_min, image_max = np.min(image), np.max(image)
                    if image_max > image_min:
                        image_display = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                    else:
                        image_display = np.zeros_like(image, dtype=np.uint8)
            else:
                image_display = image
                
            # Handle multi-band vs single-band display
            if len(image_display.shape) == 3 and image_display.shape[2] in [3, 4]:
                # RGB or RGBA image
                if image_display.shape[2] == 3:
                    pil_image = Image.fromarray(image_display, 'RGB')
                else:
                    pil_image = Image.fromarray(image_display, 'RGBA')
            elif len(image_display.shape) == 3 and image_display.shape[2] > 4:
                # More than 4 bands - use first 3 for RGB display
                image_rgb = image_display[:, :, :3]
                pil_image = Image.fromarray(image_rgb, 'RGB')
            elif len(image_display.shape) == 3 and image_display.shape[2] == 1:
                # Single band but with extra dimension
                image_squeezed = image_display.squeeze()
                pil_image = Image.fromarray(image_squeezed, 'L')
            elif len(image_display.shape) == 3:
                # Other multi-band case - convert to grayscale using mean
                image_gray = np.mean(image_display, axis=2).astype(np.uint8)
                pil_image = Image.fromarray(image_gray, 'L')
            else:
                # Single channel image
                pil_image = Image.fromarray(image_display, 'L')
                
            # Resize to fit canvas (for on-screen display only; processing stays full resolution)
            canvas.update_idletasks()  # Ensure canvas has correct dimensions
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            # Store geometry info for coordinate mapping (ROI & zoom)
            disp_w, disp_h = pil_image.size
            img_h, img_w = image.shape[:2]
            scale_x = disp_w / img_w if img_w > 0 else 1.0
            scale_y = disp_h / img_h if img_h > 0 else 1.0
            offset_x = (canvas_width - disp_w) // 2
            offset_y = (canvas_height - disp_h) // 2

            canvas._image_scale_x = scale_x
            canvas._image_scale_y = scale_y
            canvas._image_offset_x = offset_x
            canvas._image_offset_y = offset_y
            canvas._image_shape = (img_h, img_w)
            
            tk_image = ImageTk.PhotoImage(pil_image)
            canvas.image = tk_image  # Keep reference
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=tk_image)
            
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}")
            self.log(f"Image shape: {image.shape}, dtype: {image.dtype}")
            # Show error on canvas
            canvas.delete("all")
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                            text=f"Display Error\n{str(e)}", fill="red")
        
    def validate_inputs(self) -> bool:
        """Validate all user inputs."""
        try:
            kernel_size = int(self.kernel_size_var.get())
            if kernel_size < 3 or kernel_size % 2 == 0:
                messagebox.showerror("Error", "Kernel size must be odd and >= 3")
                return False
                
            sigma = float(self.sigma_var.get())
            if sigma <= 0:
                messagebox.showerror("Error", "Sigma must be positive")
                return False
                
            threshold = float(self.threshold_var.get())
            if self.threshold_mode.get() == "relative":
                if threshold < 0 or threshold > 1:
                    messagebox.showerror("Error", "Relative threshold must be between 0 and 1")
                    return False
            else:  # absolute
                if threshold < 0 or threshold > 1:
                    messagebox.showerror("Error", "Absolute threshold must be between 0 and 1")
                    return False
                
            if self.original_image is None:
                messagebox.showerror("Error", "Please load an image first")
                return False
                
            # Validate band selection for custom mode
            if self.band_var.get() == "custom":
                selected_bands = self.band_listbox.curselection()
                if not selected_bands:
                    messagebox.showerror("Error", "Please select at least one band for custom processing")
                    return False
                    
            return True
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return False
            
    def preview_processing(self):
        """Run processing on user-selected ROI (no downsampling)."""
        if not self.validate_inputs():
            return

        if self.roi is None:
            messagebox.showerror("Error", "Please select an area of interest (bounding box) on the Original tab before running preview.")
            return

        x0, y0, x1, y1 = self.roi
        preview_image = self.original_image[y0:y1, x0:x1].copy()
        
        self.abort_processing = False
        self.processing_thread = threading.Thread(target=self._process_image, 
                        args=(preview_image, True), daemon=True)
        self.processing_thread.start()
        
    def run_processing(self):
        """Run full processing on the user-selected ROI (no downsampling)."""
        if not self.validate_inputs():
            return

        if self.roi is None:
            messagebox.showerror("Error", "Please select an area of interest (bounding box) on the Original tab before running processing.")
            return

        x0, y0, x1, y1 = self.roi
        roi_image = self.original_image[y0:y1, x0:x1].copy()

        self.abort_processing = False
        self.processing_thread = threading.Thread(target=self._process_image, 
                        args=(roi_image, False), daemon=True)
        self.processing_thread.start()
        
    def abort_processing_command(self):
        """Abort the current processing operation."""
        self.abort_processing = True
        self.status_var.set("Aborting...")
        self.log("Aborting processing...")
        
    def clear_all(self):
        """Clear all results and reset the interface."""
        self.original_image = None
        self.current_image = None
        self.results = {}
        self.image_bands = []
        self.roi = None
        self._roi_rect_id = None
        
        # Clear displays
        for canvas in [self.original_canvas, self.smoothed_canvas, 
                      self.gradient_canvas, self.edges_canvas,
                      getattr(self, "box_smoothed_canvas", None),
                      getattr(self, "box_edges_canvas", None),
                      getattr(self, "gaussian_smoothed_canvas", None),
                      getattr(self, "gaussian_edges_canvas", None)]:
            if canvas is None:
                continue
            canvas.delete("all")
            canvas.image = None
            
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log_messages = []
        
        # Reset UI
        self.file_label.config(text="No file selected")
        self.metrics_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
        self.log("Cleared all results")
        
    def _process_image(self, image: np.ndarray, is_preview: bool):
        """Process image in a separate thread."""
        self.progress.start()
        self.status_var.set("Processing..." if not is_preview else "Preview processing...")
        start_time = time.time()
        
        try:
            # Get parameters
            kernel_size = int(self.kernel_size_var.get())
            sigma = float(self.sigma_var.get())
            threshold_val = float(self.threshold_var.get())
            smooth_type = self.smooth_var.get()
            band_mode = self.band_var.get()
            sobel_mode = self.sobel_var.get()
            threshold_mode = self.threshold_mode.get()
            compare = self.compare_var.get()
            
            # Log parameters
            self.log(f"Parameters: kernel={kernel_size}, sigma={sigma}, threshold={threshold_val} ({threshold_mode})")
            self.log(f"Smooth type: {smooth_type}, Band mode: {band_mode}, Sobel mode: {sobel_mode}")
            
            # Process image
            results = self.process_pipeline(
                image, kernel_size, sigma, threshold_val, 
                smooth_type, band_mode, sobel_mode, threshold_mode, compare
            )
            
            if self.abort_processing:
                self.log("Processing aborted by user")
                return
                
            processing_time = time.time() - start_time
            self.log(f"Processing completed in {processing_time:.2f} seconds")
            
            # Update UI in main thread
            self.root.after(0, self._update_results, results, is_preview)
            
        except Exception as e:
            if not self.abort_processing:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
                self.log(f"Processing error: {str(e)}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.status_var.set("Ready"))
            
    def process_pipeline(self, image: np.ndarray, kernel_size: int, sigma: float, 
                    threshold_val: float, smooth_type: str, band_mode: str,
                    sobel_mode: str, threshold_mode: str, compare: bool) -> Dict[str, Any]:
        """Execute the complete image processing pipeline."""
        results = {}
        
        # Handle band selection - with validation for multi-band images
        if band_mode == "all_bands":
            # This should only be called for images with <= 3 bands (enforced in GUI)
            if len(image.shape) == 3 and image.shape[2] > 3:
                self.log("Warning: 'All Bands' selected for multi-band image, using first 3 bands")
            working_image = process_all_bands(image)
            self.log("Processing all bands as color image (first 3 bands as RGB)")
            
        elif band_mode == "single":
            # Get selected band index
            band_desc = self.single_band_var.get()
            band_index = int(band_desc.split(':')[0])
            working_image = process_single_band(image, band_index)
            band_name = self.image_bands[band_index]['description']
            self.log(f"Using single band: {band_name} (index {band_index})")
            
        elif band_mode == "custom_rgb":
            # Get custom RGB band indices
            red_desc = self.red_band_var.get()
            green_desc = self.green_band_var.get()
            blue_desc = self.blue_band_var.get()
            
            red_idx = int(red_desc.split(':')[0])
            green_idx = int(green_desc.split(':')[0]) 
            blue_idx = int(blue_desc.split(':')[0])
            
            working_image = process_custom_rgb_bands(image, red_idx, green_idx, blue_idx)
            
            red_name = self.image_bands[red_idx]['description']
            green_name = self.image_bands[green_idx]['description']
            blue_name = self.image_bands[blue_idx]['description']
            self.log(f"Custom RGB bands - R: {red_name}, G: {green_name}, B: {blue_name}")
        else:
            working_image = image
            
        if self.abort_processing:
            return {}
        
        # Log image properties
        if len(working_image.shape) == 3:
            self.log(f"Working image: {working_image.shape[1]}x{working_image.shape[0]} with {working_image.shape[2]} bands")
        else:
            self.log(f"Working image: {working_image.shape[1]}x{working_image.shape[0]} (single channel)")
        
        # Apply smoothing
        self.log(f"Applying {smooth_type} smoothing with kernel size {kernel_size}...")
        smoothed = apply_smoothing(working_image, smooth_type, kernel_size, sigma)
        results['smoothed'] = smoothed
        
        if self.abort_processing:
            return {}
        
        # Apply Sobel
        self.log(f"Applying Sobel operator ({sobel_mode})...")
        gradient_x, gradient_y, gradient_mag = apply_sobel(smoothed, sobel_mode)
        results['gradient_x'] = gradient_x
        results['gradient_y'] = gradient_y
        results['gradient_mag'] = gradient_mag
        
        # Log gradient information
        if len(gradient_mag.shape) == 3:
            self.log(f"Multi-band gradient magnitude - shape: {gradient_mag.shape}")
            for i in range(min(3, gradient_mag.shape[2])):
                self.log(f"  Band {i}: [{np.min(gradient_mag[:,:,i]):.6f}, {np.max(gradient_mag[:,:,i]):.6f}]")
        else:
            self.log(f"Gradient magnitude range: [{np.min(gradient_mag):.6f}, {np.max(gradient_mag):.6f}]")
        
        if self.abort_processing:
            return {}
        
        # Apply threshold
        self.log(f"Applying threshold ({threshold_mode}: {threshold_val})...")
        edges = threshold_image(gradient_mag, threshold_val, threshold_mode)
        results['edges'] = edges
        results['threshold_used'] = threshold_val
        results['threshold_mode'] = threshold_mode
        
        edge_pixels = np.sum(edges > 0)
        self.log(f"Edge map: {edge_pixels} edge pixels ({edge_pixels/edges.size*100:.2f}%)")
        
        # Comparison if requested
        if compare and not self.abort_processing:
            self.log("Starting comparison between Box and Gaussian filters...")
            
            # Process with box filter
            box_smoothed = apply_smoothing(working_image, "box", kernel_size)
            box_gradient_x, box_gradient_y, box_gradient_mag = apply_sobel(box_smoothed, sobel_mode)
            box_edges = threshold_image(box_gradient_mag, threshold_val, threshold_mode)
            
            if self.abort_processing:
                return {}
            
            # Process with gaussian filter  
            gaussian_smoothed = apply_smoothing(working_image, "gaussian", kernel_size, sigma)
            gaussian_gradient_x, gaussian_gradient_y, gaussian_gradient_mag = apply_sobel(gaussian_smoothed, sobel_mode)
            gaussian_edges = threshold_image(gaussian_gradient_mag, threshold_val, threshold_mode)
            
            if self.abort_processing:
                return {}
            
            # Compare edge maps
            self.log("Computing comparison metrics...")
            comparison_metrics = compare_edge_maps(box_edges, gaussian_edges)
            
            results['comparison'] = {
                'box_smoothed': box_smoothed,
                'box_edges': box_edges,
                'gaussian_smoothed': gaussian_smoothed,
                'gaussian_edges': gaussian_edges,
                'metrics': comparison_metrics
            }
            
            self.log("Comparison completed")
            
        return results
        
    def _update_results(self, results: Dict[str, Any], is_preview: bool):
        """Update UI with processing results."""
        if self.abort_processing:
            return
            
        self.results = results
        
        # Display results
        if 'smoothed' in results:
            self.display_image(results['smoothed'], self.smoothed_canvas)
            
        if 'gradient_mag' in results:
            self.display_image(results['gradient_mag'], self.gradient_canvas)
            
        if 'edges' in results:
            self.display_image(results['edges'], self.edges_canvas)
            
        # Display comparison if available
        if 'comparison' in results:
            comp = results['comparison']

            # Keep references to full-resolution comparison images for zoom
            self.box_smoothed_image = comp.get('box_smoothed')
            self.box_edges_image = comp.get('box_edges')
            self.gaussian_smoothed_image = comp.get('gaussian_smoothed')
            self.gaussian_edges_image = comp.get('gaussian_edges')

            # Show each image separately in the comparison tab
            if self.box_smoothed_image is not None:
                self.display_image(self.box_smoothed_image, self.box_smoothed_canvas)
            if self.box_edges_image is not None:
                self.display_image(self.box_edges_image, self.box_edges_canvas)
            if self.gaussian_smoothed_image is not None:
                self.display_image(self.gaussian_smoothed_image, self.gaussian_smoothed_canvas)
            if self.gaussian_edges_image is not None:
                self.display_image(self.gaussian_edges_image, self.gaussian_edges_canvas)
            
            # Display metrics
            metrics_text = "Comparison Metrics (Box vs Gaussian):\n\n"
            metrics = comp['metrics']
            metrics_text += f"Agreement: {metrics['agreement']:.4f}\n"
            metrics_text += f"Disagreement: {metrics['disagreement']:.4f}\n"
            metrics_text += f"Precision: {metrics['precision']:.4f}\n"
            metrics_text += f"Recall: {metrics['recall']:.4f}\n"
            metrics_text += f"F1 Score: {metrics['f1_score']:.4f}\n"
            metrics_text += f"IoU: {metrics['iou']:.4f}\n"
            metrics_text += f"Hausdorff Distance: {metrics['hausdorff_distance']:.2f}\n\n"
            metrics_text += f"True Positives: {metrics['true_positives']}\n"
            metrics_text += f"False Positives: {metrics['false_positives']}\n"
            metrics_text += f"False Negatives: {metrics['false_negatives']}\n"
            metrics_text += f"True Negatives: {metrics['true_negatives']}\n"
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, metrics_text)
            
        status = "Preview" if is_preview else "Full processing"
        self.log(f"{status} completed successfully")
        
    def save_results(self):
        """Save processing results to files."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save. Run processing first.")
            return
            
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/run_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save individual results with appropriate settings
            for key, image in self.results.items():
                if key not in ['comparison', 'threshold_used', 'threshold_mode']:
                    if key == 'edges':
                        # Edge maps are binary - always save as 8-bit
                        save_image(image, f"{output_dir}/{key}.png", bit_depth=8)
                        self.log(f"Saved {key}.png (8-bit binary)")
                    elif key in ['gradient_mag', 'gradient_x', 'gradient_y']:
                        # Gradient images - check if multi-band
                        if len(image.shape) == 3 and image.shape[2] > 1:
                            # Multi-band gradients - save as 8-bit (PIL limitation for color)
                            save_image(image, f"{output_dir}/{key}.png", bit_depth=8)
                            self.log(f"Saved {key}.png (8-bit color)")
                        else:
                            # Single band gradients - save as 16-bit to preserve dynamic range
                            save_image(image, f"{output_dir}/{key}.png", bit_depth=16)
                            self.log(f"Saved {key}.png (16-bit grayscale)")
                    else:
                        # Smoothed images
                        if len(image.shape) == 3 and image.shape[2] > 1:
                            # Color smoothed images - save as 8-bit
                            save_image(image, f"{output_dir}/{key}.png", bit_depth=8)
                            self.log(f"Saved {key}.png (8-bit color)")
                        else:
                            # Single band smoothed - save as 16-bit
                            save_image(image, f"{output_dir}/{key}.png", bit_depth=16)
                            self.log(f"Saved {key}.png (16-bit grayscale)")
                    
            # Save comparison results if available
            if 'comparison' in self.results:
                comp = self.results['comparison']
                # Save smoothed and edge maps from comparison
                if 'box_smoothed' in comp:
                    save_image(comp['box_smoothed'], f"{output_dir}/box_smoothed.png", bit_depth=16)
                    self.log("Saved box_smoothed.png")
                if 'gaussian_smoothed' in comp:
                    save_image(comp['gaussian_smoothed'], f"{output_dir}/gaussian_smoothed.png", bit_depth=16)
                    self.log("Saved gaussian_smoothed.png")
                save_image(comp['box_edges'], f"{output_dir}/box_edges.png", bit_depth=8)
                save_image(comp['gaussian_edges'], f"{output_dir}/gaussian_edges.png", bit_depth=8)
                self.log("Saved comparison edge maps")
                
                # Save metrics
                import json
                with open(f"{output_dir}/comparison_metrics.json", 'w') as f:
                    json.dump(comp['metrics'], f, indent=2)
                self.log("Saved comparison metrics")
                    
            # Save processing parameters
            params = {
                'kernel_size': int(self.kernel_size_var.get()),
                'sigma': float(self.sigma_var.get()),
                'threshold': float(self.threshold_var.get()),
                'smooth_type': self.smooth_var.get(),
                'band_mode': self.band_var.get(),
                'sobel_mode': self.sobel_var.get(),
                'threshold_mode': self.threshold_mode.get(),
                'timestamp': timestamp
            }
            
            with open(f"{output_dir}/processing_parameters.json", 'w') as f:
                json.dump(params, f, indent=2)
            self.log("Saved processing parameters")
                    
            self.log(f"All results successfully saved to {output_dir}")
            messagebox.showinfo("Success", f"Results saved to {output_dir}")
            
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.log(f"Save error: {error_msg}")
            # Log more details for debugging
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            
    def export_metrics(self):
        """Export metrics to JSON file."""
        if 'comparison' not in self.results:
            messagebox.showwarning("Warning", "No comparison metrics available.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Metrics",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                import json
                metrics_data = {
                    'parameters': {
                        'kernel_size': int(self.kernel_size_var.get()),
                        'sigma': float(self.sigma_var.get()),
                        'threshold': float(self.threshold_var.get()),
                        'smooth_type': self.smooth_var.get(),
                        'band_mode': self.band_var.get(),
                        'sobel_mode': self.sobel_var.get(),
                        'threshold_mode': self.threshold_mode.get(),
                    },
                    'metrics': self.results['comparison']['metrics'],
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                self.log(f"Metrics exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export metrics: {str(e)}")

    # -----------------------
    # ROI handling on canvas
    # -----------------------
    def _canvas_to_image_coords(self, canvas: tk.Canvas, x: int, y: int) -> Tuple[int, int]:
        """Map canvas coordinates to image coordinates using stored scale/offset."""
        scale_x = getattr(canvas, "_image_scale_x", 1.0)
        scale_y = getattr(canvas, "_image_scale_y", 1.0)
        offset_x = getattr(canvas, "_image_offset_x", 0)
        offset_y = getattr(canvas, "_image_offset_y", 0)
        img_h, img_w = getattr(canvas, "_image_shape", (0, 0))

        ix = int((x - offset_x) / scale_x)
        iy = int((y - offset_y) / scale_y)

        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))
        return ix, iy

    def _on_original_mouse_down(self, event: tk.Event):
        """Start ROI selection on the original image canvas."""
        if self.original_image is None:
            return
        self._roi_canvas_start = (event.x, event.y)
        if self._roi_rect_id is not None:
            self.original_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _on_original_mouse_drag(self, event: tk.Event):
        """Update ROI rectangle while dragging."""
        if self._roi_canvas_start is None:
            return
        x0, y0 = self._roi_canvas_start
        x1, y1 = event.x, event.y
        if self._roi_rect_id is None:
            self._roi_rect_id = self.original_canvas.create_rectangle(
                x0, y0, x1, y1, outline="red", width=2
            )
        else:
            self.original_canvas.coords(self._roi_rect_id, x0, y0, x1, y1)

    def _on_original_mouse_up(self, event: tk.Event):
        """Finalize ROI selection and convert to image coordinates."""
        if self._roi_canvas_start is None or self.original_image is None:
            return

        x0_c, y0_c = self._roi_canvas_start
        x1_c, y1_c = event.x, event.y

        # Normalize canvas coordinates
        x0_c, x1_c = sorted((x0_c, x1_c))
        y0_c, y1_c = sorted((y0_c, y1_c))

        # Map to image coordinates
        x0_i, y0_i = self._canvas_to_image_coords(self.original_canvas, x0_c, y0_c)
        x1_i, y1_i = self._canvas_to_image_coords(self.original_canvas, x1_c, y1_c)

        # Ensure non-zero size ROI
        if x1_i <= x0_i or y1_i <= y0_i:
            self.log("Ignored ROI with zero size")
            return

        self.roi = (x0_i, y0_i, x1_i, y1_i)
        self.log(f"Selected ROI: x=({x0_i}, {x1_i}), y=({y0_i}, {y1_i})")

    # -----------------------
    # Zoom handling (comparison tab)
    # -----------------------
    def _apply_zoom_to_canvas(self, canvas: tk.Canvas, image: Optional[np.ndarray], ix: int, iy: int):
        """Apply 1x1 zoom around (ix, iy) to a single canvas."""
        if image is None:
            return

        img_h, img_w = image.shape[:2]
        half_w = min(128, img_w // 2)
        half_h = min(128, img_h // 2)

        x0 = max(0, ix - half_w)
        x1 = min(img_w, ix + half_w)
        y0 = max(0, iy - half_h)
        y1 = min(img_h, iy + half_h)

        zoom_region = image[y0:y1, x0:x1].copy()

        # Normalize for display (no resizing -> 1 image pixel == 1 screen pixel)
        if zoom_region.dtype != np.uint8:
            if np.max(zoom_region) <= 1.0 and np.min(zoom_region) >= 0:
                zoom_display = (zoom_region * 255).astype(np.uint8)
            else:
                zmin, zmax = np.min(zoom_region), np.max(zoom_region)
                if zmax > zmin:
                    zoom_display = ((zoom_region - zmin) / (zmax - zmin) * 255).astype(np.uint8)
                else:
                    zoom_display = np.zeros_like(zoom_region, dtype=np.uint8)
        else:
            zoom_display = zoom_region

        if len(zoom_display.shape) == 3 and zoom_display.shape[2] in [3, 4]:
            mode = 'RGB' if zoom_display.shape[2] == 3 else 'RGBA'
            pil_zoom = Image.fromarray(zoom_display, mode)
        elif len(zoom_display.shape) == 3 and zoom_display.shape[2] > 4:
            pil_zoom = Image.fromarray(zoom_display[:, :, :3], 'RGB')
        elif len(zoom_display.shape) == 3 and zoom_display.shape[2] == 1:
            pil_zoom = Image.fromarray(zoom_display.squeeze(), 'L')
        elif len(zoom_display.shape) == 3:
            gray = np.mean(zoom_display, axis=2).astype(np.uint8)
            pil_zoom = Image.fromarray(gray, 'L')
        else:
            pil_zoom = Image.fromarray(zoom_display, 'L')

        # Draw zoomed image centered in the same canvas (no scaling)
        canvas.update_idletasks()
        c_w = canvas.winfo_width()
        c_h = canvas.winfo_height()
        tk_zoom_img = ImageTk.PhotoImage(pil_zoom)
        canvas.image = tk_zoom_img
        canvas.delete("all")
        canvas.create_image(c_w // 2, c_h // 2, anchor=tk.CENTER, image=tk_zoom_img)

    def _on_comparison_zoom(self, source_canvas: tk.Canvas, event: tk.Event):
        """Handle synchronized zoom across all four comparison images."""
        if (
            self.box_smoothed_image is None
            or self.box_edges_image is None
            or self.gaussian_smoothed_image is None
            or self.gaussian_edges_image is None
        ):
            return

        # Compute image coordinates from the canvas that was clicked
        ix, iy = self._canvas_to_image_coords(source_canvas, event.x, event.y)

        # Apply the same (ix, iy) zoom to all four images/canvases
        self._apply_zoom_to_canvas(self.box_smoothed_canvas, self.box_smoothed_image, ix, iy)
        self._apply_zoom_to_canvas(self.box_edges_canvas, self.box_edges_image, ix, iy)
        self._apply_zoom_to_canvas(self.gaussian_smoothed_canvas, self.gaussian_smoothed_image, ix, iy)
        self._apply_zoom_to_canvas(self.gaussian_edges_canvas, self.gaussian_edges_image, ix, iy)

    def _reset_comparison_zoom(self):
        """Reset all comparison canvases back to their full image views."""
        if self.box_smoothed_image is not None:
            self.display_image(self.box_smoothed_image, self.box_smoothed_canvas)
        if self.box_edges_image is not None:
            self.display_image(self.box_edges_image, self.box_edges_canvas)
        if self.gaussian_smoothed_image is not None:
            self.display_image(self.gaussian_smoothed_image, self.gaussian_smoothed_canvas)
        if self.gaussian_edges_image is not None:
            self.display_image(self.gaussian_edges_image, self.gaussian_edges_canvas)
