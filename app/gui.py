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

from app.io_utils import load_image, save_image
from app.processing import (
    apply_smoothing, 
    apply_sobel, 
    threshold_image,
    generate_box_kernel,
    generate_gaussian_kernel
)
from app.viz import create_comparison_display
from app.metrics import compare_edge_maps

class ImageProcessingApp:
    """Main application GUI for image edge processing."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Edge Processor")
        self.root.geometry("1200x800")
        
        # State variables
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.results: Dict[str, Any] = {}
        self.setup_logging()
        self.create_widgets()
        
    def setup_logging(self):
        """Setup logging to text widget."""
        self.log_messages: List[str] = []
        
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
        # File selection
        file_frame = ttk.Frame(parent)
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", 
                  command=self.load_image).grid(row=0, column=0, sticky=tk.W)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        file_frame.columnconfigure(1, weight=1)
        
        # Smoothing options
        smooth_frame = ttk.LabelFrame(parent, text="Smoothing", padding="5")
        smooth_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
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
        
        # Band selection
        band_frame = ttk.LabelFrame(parent, text="Band Processing", padding="5")
        band_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.band_var = tk.StringVar(value="grayscale")
        ttk.Radiobutton(band_frame, text="Grayscale (Luminance)", 
                       variable=self.band_var, value="grayscale").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(band_frame, text="All Bands Separate", 
                       variable=self.band_var, value="all").grid(row=1, column=0, sticky=tk.W)
        
        # Sobel options
        sobel_frame = ttk.LabelFrame(parent, text="Sobel Operator", padding="5")
        sobel_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.sobel_var = tk.StringVar(value="both")
        ttk.Radiobutton(sobel_frame, text="X Direction", 
                       variable=self.sobel_var, value="x").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(sobel_frame, text="Y Direction", 
                       variable=self.sobel_var, value="y").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(sobel_frame, text="Both (Magnitude)", 
                       variable=self.sobel_var, value="both").grid(row=2, column=0, sticky=tk.W)
        
        # Threshold options
        threshold_frame = ttk.LabelFrame(parent, text="Threshold", padding="5")
        threshold_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.2)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        self.threshold_mode = tk.StringVar(value="relative")
        ttk.Radiobutton(threshold_frame, text="Relative", 
                       variable=self.threshold_mode, value="relative").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(threshold_frame, text="Absolute", 
                       variable=self.threshold_mode, value="absolute").grid(row=2, column=1, sticky=tk.W)
        
        # Comparison toggle
        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Compare Box vs Gaussian", 
                       variable=self.compare_var).grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        
        # Action buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Preview", 
                  command=self.preview_processing).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Run Full Processing", 
                  command=self.run_processing).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Export Metrics", 
                  command=self.export_metrics).grid(row=0, column=3, padx=(5, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
        
        self.comparison_canvas = tk.Canvas(self.comparison_tab, bg='white')
        self.comparison_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Metrics display
        self.metrics_text = tk.Text(self.comparison_tab, height=10, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=False)
        
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
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image = load_image(file_path)
                self.current_image = self.original_image.copy()
                self.file_label.config(text=os.path.basename(file_path))
                self.display_image(self.original_image, self.original_canvas)
                self.log(f"Loaded image: {file_path}, shape: {self.original_image.shape}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.log(f"Error loading image: {str(e)}")
                
    def display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """Display an image on a canvas."""
        if image is None:
            return
            
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image_display = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_display = image
            
        if len(image_display.shape) == 3 and image_display.shape[2] == 3:
            pil_image = Image.fromarray(image_display, 'RGB')
        else:
            pil_image = Image.fromarray(image_display.squeeze(), 'L')
            
        # Resize to fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas.image = tk_image  # Keep reference
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=tk_image)
        
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
            if threshold < 0 or threshold > 1:
                messagebox.showerror("Error", "Threshold must be between 0 and 1")
                return False
                
            if self.original_image is None:
                messagebox.showerror("Error", "Please load an image first")
                return False
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return False
            
    def preview_processing(self):
        """Run processing on a downsampled preview."""
        if not self.validate_inputs():
            return
            
        # Create downsampled version for preview
        preview_image = self.original_image[::2, ::2]  # Simple downsampling
        
        threading.Thread(target=self._process_image, 
                        args=(preview_image, True), daemon=True).start()
        
    def run_processing(self):
        """Run full processing on the original image."""
        if not self.validate_inputs():
            return
            
        threading.Thread(target=self._process_image, 
                        args=(self.original_image, False), daemon=True).start()
        
    def _process_image(self, image: np.ndarray, is_preview: bool):
        """Process image in a separate thread."""
        self.progress.start()
        self.log(f"Starting {'preview' if is_preview else 'full'} processing...")
        
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
            self.log(f"Parameters: kernel={kernel_size}, sigma={sigma}, threshold={threshold_val}")
            self.log(f"Smooth type: {smooth_type}, Band mode: {band_mode}")
            
            # Process image
            results = self.process_pipeline(
                image, kernel_size, sigma, threshold_val, 
                smooth_type, band_mode, sobel_mode, threshold_mode, compare
            )
            
            # Update UI in main thread
            self.root.after(0, self._update_results, results, is_preview)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.log(f"Processing error: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, self.progress.stop)
            
    def process_pipeline(self, image: np.ndarray, kernel_size: int, sigma: float, 
                        threshold_val: float, smooth_type: str, band_mode: str,
                        sobel_mode: str, threshold_mode: str, compare: bool) -> Dict[str, Any]:
        """Execute the complete image processing pipeline."""
        results = {}
        
        # Handle band selection
        if band_mode == "grayscale" and len(image.shape) == 3:
            # Convert to grayscale using luminance formula
            if image.shape[2] == 3:  # RGB
                gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
            else:  # Multi-band, use first channel
                gray = image[:,:,0]
            working_image = gray
        else:
            working_image = image
            
        # Apply smoothing
        if smooth_type == "box":
            smoothed = apply_smoothing(working_image, "box", kernel_size)
        else:  # gaussian
            smoothed = apply_smoothing(working_image, "gaussian", kernel_size, sigma)
            
        results['smoothed'] = smoothed
        
        # Apply Sobel
        gradient_x, gradient_y, gradient_mag = apply_sobel(smoothed, sobel_mode)
        results['gradient_x'] = gradient_x
        results['gradient_y'] = gradient_y
        results['gradient_mag'] = gradient_mag
        
        # Apply threshold
        if threshold_mode == "relative":
            absolute_threshold = threshold_val * np.max(gradient_mag)
        else:
            absolute_threshold = threshold_val
            
        edges = threshold_image(gradient_mag, absolute_threshold)
        results['edges'] = edges
        results['threshold_used'] = absolute_threshold
        
        # Comparison if requested
        if compare:
            # Process with box filter
            box_smoothed = apply_smoothing(working_image, "box", kernel_size)
            box_gradient_x, box_gradient_y, box_gradient_mag = apply_sobel(box_smoothed, sobel_mode)
            box_edges = threshold_image(box_gradient_mag, absolute_threshold)
            
            # Process with gaussian filter
            gaussian_smoothed = apply_smoothing(working_image, "gaussian", kernel_size, sigma)
            gaussian_gradient_x, gaussian_gradient_y, gaussian_gradient_mag = apply_sobel(gaussian_smoothed, sobel_mode)
            gaussian_edges = threshold_image(gaussian_gradient_mag, absolute_threshold)
            
            # Compare edge maps
            comparison_metrics = compare_edge_maps(box_edges, gaussian_edges)
            
            results['comparison'] = {
                'box_edges': box_edges,
                'gaussian_edges': gaussian_edges,
                'metrics': comparison_metrics
            }
            
        return results
        
    def _update_results(self, results: Dict[str, Any], is_preview: bool):
        """Update UI with processing results."""
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
            # Create comparison display
            comparison_image = create_comparison_display(
                comp['box_edges'], comp['gaussian_edges']
            )
            self.display_image(comparison_image, self.comparison_canvas)
            
            # Display metrics
            metrics_text = "Comparison Metrics:\n\n"
            for key, value in comp['metrics'].items():
                metrics_text += f"{key}: {value:.4f}\n"
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
            # Save individual results
            for key, image in self.results.items():
                if key != 'comparison' and key != 'threshold_used':
                    save_image(image, f"{output_dir}/{key}.png")
                    
            # Save comparison results if available
            if 'comparison' in self.results:
                comp = self.results['comparison']
                save_image(comp['box_edges'], f"{output_dir}/box_edges.png")
                save_image(comp['gaussian_edges'], f"{output_dir}/gaussian_edges.png")
                
                # Save metrics
                import json
                with open(f"{output_dir}/comparison_metrics.json", 'w') as f:
                    json.dump(comp['metrics'], f, indent=2)
                    
            self.log(f"Results saved to {output_dir}")
            messagebox.showinfo("Success", f"Results saved to {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            self.log(f"Save error: {str(e)}")
            
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
                with open(file_path, 'w') as f:
                    json.dump(self.results['comparison']['metrics'], f, indent=2)
                self.log(f"Metrics exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export metrics: {str(e)}")