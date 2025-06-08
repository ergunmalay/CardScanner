"""
YOLO CardMarket Scanner - Main Scanner Module
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from PIL import Image, ImageTk
import time
import json
import os
import re  # Added missing import
from difflib import SequenceMatcher
from ultralytics import YOLO

# Import OCR handler
from ocr_handler import OCRHandler

# Import utilities
from utils import create_price_overlay_image, calculate_similarity

class YOLOCardScanner:
    def __init__(self, typhoon_available=False, tesseract_available=False):
        # Initialize OCR handler
        self.ocr_handler = OCRHandler(typhoon_available, tesseract_available)
        
        # Model and camera
        self.model = None
        self.cap = None
        self.is_scanning = False
        
        # Data storage
        self.singles_data = []
        self.prices_data = None
        
        # Detection state
        self.last_card_roi = None
        self.last_capture_time = 0
        self.captured_cards = []
        
        # Display state
        self.captured_card_image = None
        self.captured_ocr_display = None
        self.processed_ocr_image = None
        self.last_found_price = None
        
        # Setup GUI
        self.setup_gui()
        
        # Load data and model
        self.load_data_files()
        self.load_model()
        self.create_captures_folder()
    
    def setup_gui(self):
        """Setup GUI with 4 display panels"""
        self.root = tk.Tk()
        self.root.title("YOLO CardMarket Scanner - 4 View Display")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLO CardMarket Scanner", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=5)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Model controls
        model_frame = ttk.LabelFrame(control_frame, text="Model & Data", padding="5")
        model_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_frame, text="Load Model", 
                  command=self.load_model_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_frame, text="Reload Data", 
                  command=self.load_data_files).pack(side=tk.LEFT, padx=2)
        
        self.model_status = ttk.Label(model_frame, text="No model")
        self.model_status.pack(side=tk.LEFT, padx=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        camera_frame.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(camera_frame, text="Start", 
                                   command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.auto_scan_var = tk.BooleanVar(value=True)
        self.auto_scan_cb = ttk.Checkbutton(camera_frame, text="Auto Capture", 
                                           variable=self.auto_scan_var)
        self.auto_scan_cb.pack(side=tk.LEFT, padx=2)
        
        self.manual_scan_btn = ttk.Button(camera_frame, text="Manual Scan", 
                                         command=self.manual_scan, state='disabled')
        self.manual_scan_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop", 
                                  command=self.stop_camera, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # OCR settings
        ocr_frame = ttk.LabelFrame(control_frame, text="OCR Engine", padding="5")
        ocr_frame.pack(side=tk.LEFT, padx=5)
        
        self.ocr_engine = tk.StringVar(value=self.ocr_handler.default_engine)
        
        if self.ocr_handler.typhoon_available:
            ttk.Radiobutton(ocr_frame, text="Typhoon OCR", variable=self.ocr_engine, 
                           value="typhoon").pack(side=tk.LEFT, padx=2)
        else:
            ttk.Label(ocr_frame, text="Typhoon: Not Available", 
                     foreground="gray").pack(side=tk.LEFT, padx=2)
        
        if self.ocr_handler.tesseract_available:
            ttk.Radiobutton(ocr_frame, text="Tesseract", variable=self.ocr_engine, 
                           value="tesseract").pack(side=tk.LEFT, padx=2)
        else:
            ttk.Label(ocr_frame, text="Tesseract: Not Available", 
                     foreground="gray").pack(side=tk.LEFT, padx=2)
        
        # Capture settings
        capture_frame = ttk.LabelFrame(control_frame, text="Capture Settings", padding="5")
        capture_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(capture_frame, text="Min Confidence:").pack(side=tk.LEFT, padx=2)
        self.confidence_var = tk.StringVar(value="0.6")
        confidence_spin = ttk.Spinbox(capture_frame, from_=0.1, to=1.0, width=5,
                                     increment=0.1, textvariable=self.confidence_var)
        confidence_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(capture_frame, text="View Captures", 
                  command=self.show_captures).pack(side=tk.LEFT, padx=2)
        
        # Manual search section
        search_frame = ttk.LabelFrame(control_frame, text="Manual Search", padding="5")
        search_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(search_frame, text="Card Name:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=2)
        search_entry.bind('<Return>', lambda e: self.manual_search())
        
        ttk.Button(search_frame, text="Search", 
                  command=self.manual_search).pack(side=tk.LEFT, padx=2)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready", 
                                     font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # 4-Panel Display Area
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Top row - Detection and Transform
        top_row = ttk.Frame(display_frame)
        top_row.pack(fill=tk.BOTH, expand=True)
        
        # Panel 1: YOLO Detection
        detection_panel = ttk.LabelFrame(top_row, text="1. YOLO Detection", padding="5")
        detection_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.detection_label = ttk.Label(detection_panel, text="Camera feed will appear here",
                                        background="black", foreground="white", anchor="center")
        self.detection_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel 2: Card Transform
        transform_panel = ttk.LabelFrame(top_row, text="2. Card Transform", padding="5")
        transform_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.transform_label = ttk.Label(transform_panel, text="Extracted card will appear here",
                                        background="black", foreground="white", anchor="center")
        self.transform_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom row - OCR and Results
        bottom_row = ttk.Frame(display_frame)
        bottom_row.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel 3: OCR Processing with Price
        ocr_panel = ttk.LabelFrame(bottom_row, text="3. OCR Processed Image + Price", padding="5")
        ocr_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.ocr_label = ttk.Label(ocr_panel, text="OCR processed image will appear here",
                                  background="black", foreground="white", anchor="center")
        self.ocr_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel 4: Results
        results_panel = ttk.LabelFrame(bottom_row, text="4. OCR Results & Prices", padding="5")
        results_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.results_text = tk.Text(results_panel, height=15, width=50, wrap=tk.WORD,
                                   background="black", foreground="green", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(results_panel, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial log messages
        self.add_log("🚀 YOLO CardMarket Scanner Ready")
        self.add_log("📁 Loading data files...")
        self.add_log("📸 Auto-capture enabled for 60%+ confidence detections")
        self.add_log("🔄 Version 2.0 - Modular Architecture")
    
    def create_captures_folder(self):
        """Create folder for captured cards"""
        if not os.path.exists("captured_cards"):
            os.makedirs("captured_cards")
            self.add_log("📁 Created captured_cards folder")
    
    def load_model(self):
        """Load default YOLO model"""
        try:
            model_path = "runs/detect/train/weights/best.pt"
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_status.config(text="✅ Model loaded")
                self.add_log("✅ YOLO model loaded")
            else:
                self.model_status.config(text="❌ No model found")
                self.add_log("❌ Default model not found")
        except Exception as e:
            self.add_log(f"❌ Model error: {e}")
    
    def load_model_file(self):
        """Load custom YOLO model"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model (.pt)",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.model = YOLO(file_path)
            self.model_status.config(text=f"✅ {os.path.basename(file_path)}")
            self.add_log(f"✅ Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def load_data_files(self):
        """Load merged JSON data file"""
        merged_file = "cards_with_prices.json"
        
        # Check if merged file exists
        if not os.path.exists(merged_file):
            self.add_log("❌ cards_with_prices.json not found!")
            self.add_log("💡 Please run merge_json_files.py first to create the merged data file")
            self.add_log("   Command: python merge_json_files.py")
            self.singles_data = []
            self.prices_data = None
            return
        
        # Load merged file
        self.add_log("📂 Loading cards_with_prices.json...")
        try:
            with open(merged_file, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            if isinstance(merged_data, dict) and 'cards' in merged_data:
                self.singles_data = merged_data['cards']
                self.prices_data = None  # Not needed with merged data
                
                count = len(self.singles_data) if isinstance(self.singles_data, list) else 0
                self.add_log(f"✅ Loaded {count} cards from merged file")
                
                # Count cards with prices
                cards_with_prices = sum(1 for card in self.singles_data if card.get('prices', {}))
                self.add_log(f"💰 Cards with price data: {cards_with_prices}/{count}")
                
                # Show sample entries
                if self.singles_data and len(self.singles_data) > 0:
                    self.add_log("📄 Sample cards:")
                    for i, card in enumerate(self.singles_data[:3]):
                        name = card.get('name', 'Unknown')
                        prices = card.get('prices', {})
                        price_fields = list(prices.keys())[:3]
                        self.add_log(f"  {i+1}. '{name}' - Prices: {price_fields}")
                
                self.add_log("✅ Data loaded - ready to scan!")
            else:
                self.add_log("❌ Invalid merged file structure")
                self.add_log("💡 Please regenerate the file with merge_json_files.py")
                self.singles_data = []
                self.prices_data = None
                
        except json.JSONDecodeError as e:
            self.add_log(f"❌ Invalid JSON in merged file: {e}")
            self.singles_data = []
            self.prices_data = None
        except Exception as e:
            self.add_log(f"❌ Error loading merged file: {e}")
            self.singles_data = []
            self.prices_data = None
    
    def start_camera(self):
        """Start camera"""
        if not self.model:
            messagebox.showerror("Error", "Load YOLO model first!")
            return
            
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
            
            self.is_scanning = True
            self.start_btn.config(state='disabled')
            self.manual_scan_btn.config(state='normal')
            self.stop_btn.config(state='normal')
            
            self.status_label.config(text="🔴 SCANNING...")
            self.add_log("🎥 Camera started")
            
            # Start detection loop
            threading.Thread(target=self.detection_loop, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {e}")
    
    def detection_loop(self):
        """Main detection loop"""
        while self.is_scanning and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            try:
                # 1. YOLO Detection
                results = self.model(frame, verbose=False)
                detection_frame = results[0].plot()
                
                # Update Panel 1: Detection
                self.update_panel_image(self.detection_label, detection_frame, (320, 240))
                
                # Process detections
                detections = results[0].boxes
                
                if detections is not None and len(detections) > 0:
                    best_detection = detections[0]
                    confidence = float(best_detection.conf[0])
                    
                    # Get confidence threshold
                    try:
                        min_confidence = float(self.confidence_var.get())
                    except:
                        min_confidence = 0.6
                    
                    if confidence >= min_confidence:
                        # 2. Extract and transform card
                        transformed_card = self.extract_and_transform_card(frame, best_detection)
                        
                        if transformed_card is not None:
                            # Store current detection for manual scan
                            self.last_card_roi = transformed_card
                            
                            # 3. Auto-capture logic
                            current_time = time.time()
                            should_capture = (self.auto_scan_var.get() and 
                                            current_time - self.last_capture_time > 2.0)
                            
                            if should_capture:
                                # Capture new card
                                self.capture_and_process_card(transformed_card, confidence)
                                self.last_capture_time = current_time
                
                # Always display captured card in transform panel
                if self.captured_card_image is not None:
                    self.update_panel_image(self.transform_label, self.captured_card_image, (300, 400))
                else:
                    self.show_empty_panel(self.transform_label, "No card captured yet", (300, 400))
                
                # Always display processed OCR image with price overlay in OCR panel
                if self.processed_ocr_image is not None:
                    display_image = create_price_overlay_image(self.processed_ocr_image, self.last_found_price)
                    self.update_panel_image(self.ocr_label, display_image, (400, 400))
                else:
                    self.show_empty_panel(self.ocr_label, "Waiting for capture...", (400, 400))
                
            except Exception as e:
                self.add_log(f"❌ Detection error: {e}")
                break
    
    def update_panel_image(self, label, cv_image, size):
        """Update a panel with OpenCV image"""
        try:
            # Resize image
            resized = cv2.resize(cv_image, size)
            
            # Convert to RGB
            if len(resized.shape) == 3:
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL and then to PhotoImage
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            label.configure(image=photo, text="")
            label.image = photo
            
        except Exception as e:
            pass
    
    def show_empty_panel(self, label, text, size):
        """Show empty panel with text"""
        try:
            empty = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.putText(empty, text, (20, size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            self.update_panel_image(label, empty, size)
        except Exception as e:
            pass
    
    def extract_and_transform_card(self, frame, detection):
        """Extract and transform card from detection"""
        try:
            # Get bounding box
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            
            # Add padding
            padding = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract card region
            card_roi = frame[y1:y2, x1:x2]
            
            if card_roi.size == 0:
                return None
            
            # Simple resize
            target_width = 300
            target_height = 420
            transformed = cv2.resize(card_roi, (target_width, target_height))
            
            return transformed
            
        except Exception as e:
            return None
    
    def crop_bottom_half_for_ocr(self, card_image):
        """Crop bottom half of the card image for OCR processing"""
        try:
            height = card_image.shape[0]
            # Keep only top half (remove bottom half)
            cropped = card_image[0:height//2, :]
            
            self.add_log(f"🔄 Cropped image: {card_image.shape} → {cropped.shape}")
            return cropped
            
        except Exception as e:
            self.add_log(f"❌ Crop error: {e}")
            return card_image
    
    def capture_and_process_card(self, card_image, confidence):
        """Capture card image and perform OCR analysis"""
        try:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captured_cards/card_{timestamp}_conf{confidence:.2f}.jpg"
            
            # Save the captured card
            cv2.imwrite(filename, card_image)
            
            # Store captured card for persistent display
            self.captured_card_image = card_image.copy()
            
            self.add_log(f"📸 CAPTURED: {filename} (confidence: {confidence:.3f})")
            
            # Crop bottom half for OCR processing
            cropped_for_ocr = self.crop_bottom_half_for_ocr(card_image)
            
            # Store the processed image for display
            self.processed_ocr_image = cropped_for_ocr.copy()
            
            # Perform OCR analysis on cropped image
            selected_engine = self.ocr_engine.get()
            ocr_results = self.ocr_handler.perform_ocr(cropped_for_ocr, selected_engine)
            
            # Store capture info
            capture_info = {
                'filename': filename,
                'timestamp': timestamp,
                'confidence': confidence,
                'ocr_results': ocr_results
            }
            self.captured_cards.append(capture_info)
            
            # Display full OCR text in results panel
            self.display_full_ocr_results(ocr_results, timestamp)
            
            # Perform price lookup if text detected
            if ocr_results.get('best_text'):
                self.perform_price_lookup(ocr_results['best_text'])
            
        except Exception as e:
            self.add_log(f"❌ Capture error: {e}")
    
    def display_full_ocr_results(self, ocr_results, timestamp):
        """Display full OCR results in the results panel"""
        try:
            self.add_log(f"\n🔍 OCR RESULTS ({timestamp}):")
            self.add_log("=" * 40)
            
            engine = ocr_results.get('engine', 'unknown')
            self.add_log(f"Engine: {engine.upper()}")
            
            if ocr_results.get('confidence'):
                self.add_log(f"Confidence: {ocr_results['confidence']}%")
            
            full_text = ocr_results.get('full_text', '')
            if full_text and full_text != "No OCR engine available":
                self.add_log("\n📄 FULL OCR TEXT:")
                self.add_log("-" * 20)
                
                # Split into lines and display each
                lines = full_text.split('\n')
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        self.add_log(f"{i:2d}: {line.strip()}")
                
                self.add_log("-" * 20)
                
                # Extract potential card name for price lookup
                card_name = self.extract_card_name_from_full_text(full_text)
                if card_name:
                    self.add_log(f"🎯 Extracted Card Name: {card_name}")
                    
            else:
                self.add_log("❌ No text detected")
            
            self.add_log("=" * 40)
            
        except Exception as e:
            self.add_log(f"❌ Display error: {e}")
    
    def extract_card_name_from_full_text(self, full_text):
        """Extract likely card name from full OCR text for price lookup"""
        try:
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            if not lines:
                return ""
            
            # Look for title patterns
            for line in lines:
                line_lower = line.lower()
                
                # Skip description-like lines
                skip_words = ['action', 'attack', 'defense', 'once per turn', 'when', 'if', 'target', 
                            'deal', 'prevent', 'destroy', 'goes again', 'cost', 'combat chain']
                
                if any(skip in line_lower for skip in skip_words):
                    continue
                
                # Look for known card names or title-like patterns
                if len(line.split()) <= 6 and len(line) >= 3:
                    # Clean the line
                    cleaned = re.sub(r'[^\w\s]', ' ', line)
                    cleaned = ' '.join(cleaned.split())
                    
                    if cleaned:
                        return cleaned
            
            # Fallback: return first non-empty line
            if lines:
                first_line = re.sub(r'[^\w\s]', ' ', lines[0])
                first_line = ' '.join(first_line.split()[:4])  # First 4 words
                return first_line
            
            return ""
            
        except Exception as e:
            return ""
    
    def manual_search(self):
        """Manual search for testing"""
        search_term = self.search_var.get().strip()
        if not search_term:
            self.add_log("⚠️  Enter a card name to search")
            return
        
        self.add_log(f"\n🔍 MANUAL SEARCH: '{search_term}'")
        self.perform_price_lookup(search_term)
    
    def manual_scan(self):
        """Manual scan of current detection"""
        if self.last_card_roi is not None:
            self.capture_and_process_card(self.last_card_roi, 1.0)
        else:
            self.add_log("⚠️  No card detected for manual scan")
    
    def perform_price_lookup(self, card_name):
        """Perform price lookup using merged data"""
        try:
            self.add_log(f"\n🔍 PRICE LOOKUP: {card_name}")
            
            if not self.singles_data:
                self.add_log("❌ No card data loaded")
                self.add_log("💡 Run merge_json_files.py to create cards_with_prices.json")
                return
            
            # Search in merged database
            self.add_log("🔍 Searching in merged card database...")
            card_match = self.find_card_with_price(card_name)
            
            if not card_match:
                self.add_log("❌ No matches found in card database")
                return
            
            self.add_log(f"✅ Found card: {card_match['name']}")
            self.add_log(f"🆔 Product ID: {card_match['product_id']}")
            self.add_log(f"📊 Match confidence: {card_match['confidence']:.1%}")
            
            # Extract price from embedded data
            price_info = self.extract_price_from_card(card_match)
            
            if price_info:
                self.add_log(f"💰 PRICE FOUND: {price_info}")
                
                # Store price info for persistent display
                self.last_found_price = {
                    'name': card_match['name'],
                    'price': price_info,
                    'confidence': card_match['confidence'],
                    'product_id': card_match['product_id']
                }
                
                self.add_log("🎯 Price will persist in OCR panel until next card found")
            else:
                self.add_log("❌ No price data found for this card")
                
                # Store the card info anyway but with "No Price" indicator
                self.last_found_price = {
                    'name': card_match['name'],
                    'price': "No Price Found",
                    'confidence': card_match['confidence'],
                    'product_id': card_match['product_id']
                }
                
                self.add_log("🎯 Card found but no price - will show in OCR panel")
                
        except Exception as e:
            self.add_log(f"❌ Lookup error: {e}")
            import traceback
            self.add_log(f"❌ Full traceback: {traceback.format_exc()}")
    
    def find_card_with_price(self, card_name):
        """Find card in merged database"""
        try:
            target_name = card_name.strip().lower()
            self.add_log(f"🔍 Searching for: '{target_name}' in {len(self.singles_data)} cards")
            
            best_match = None
            best_score = 0
            all_matches = []
            
            # Search through all cards
            for item in self.singles_data:
                if isinstance(item, dict) and 'name' in item:
                    item_name = str(item['name'])
                    item_name_lower = item_name.lower()
                    
                    # Multiple matching strategies
                    scores = []
                    
                    # 1. Exact match
                    if target_name == item_name_lower:
                        scores.append(1.0)
                    
                    # 2. Contains match
                    if target_name in item_name_lower or item_name_lower in target_name:
                        scores.append(0.9)
                    
                    # 3. Fuzzy similarity
                    fuzzy_score = calculate_similarity(target_name, item_name_lower)
                    scores.append(fuzzy_score)
                    
                    # 4. Word-by-word matching
                    target_words = set(target_name.split())
                    item_words = set(item_name_lower.split())
                    if target_words and item_words:
                        word_match = len(target_words.intersection(item_words)) / len(target_words.union(item_words))
                        scores.append(word_match)
                    
                    # Take the best score
                    final_score = max(scores) if scores else 0
                    
                    if final_score > 0.5:  # Higher threshold for better matches
                        match_info = {
                            'name': item_name,
                            'product_id': item.get('idProduct'),
                            'confidence': final_score,
                            'full_item': item,
                            'prices': item.get('prices', {})  # Embedded price data
                        }
                        all_matches.append(match_info)
                        
                        if final_score > best_score:
                            best_score = final_score
                            best_match = match_info
            
            # Show matches for debugging
            if all_matches:
                self.add_log(f"🎯 Found {len(all_matches)} potential matches:")
                for i, match in enumerate(sorted(all_matches, key=lambda x: x['confidence'], reverse=True)[:3]):
                    has_prices = bool(match['prices'])
                    price_fields = list(match['prices'].keys())[:3] if match['prices'] else []
                    self.add_log(f"  {i+1}. '{match['name']}' - Score: {match['confidence']:.3f} - ID: {match['product_id']} - Prices: {price_fields}")
            else:
                self.add_log("❌ No matches found")
            
            return best_match
            
        except Exception as e:
            self.add_log(f"❌ Card search error: {e}")
            return None
    
    def extract_price_from_card(self, card_match):
        """Extract price from merged card data"""
        try:
            prices = card_match.get('prices', {})
            
            if not prices:
                self.add_log("❌ No price data in card")
                return None
            
            self.add_log(f"🔍 Available price fields: {list(prices.keys())}")
            
            # Try different price fields in order of preference
            price_fields = [
                ('Low', 'low'),
                ('Avg', 'avg'),  
                ('Trend', 'trend'),
                ('Avg1', 'avg1'),
                ('Avg7', 'avg7'),
                ('Avg30', 'avg30')
            ]
            
            for label, field in price_fields:
                if field in prices and prices[field] is not None:
                    try:
                        value = prices[field]
                        self.add_log(f"🔍 Checking field '{field}': {value} (type: {type(value)})")
                        
                        # Handle different value types
                        if isinstance(value, (int, float)) and value > 0:
                            price = float(value)
                            self.add_log(f"💰 Found valid price: {field} = €{price:.2f}")
                            return f"€{price:.2f} ({label})"
                        elif isinstance(value, str):
                            clean = value.replace('€', '').replace(',', '').strip()
                            if clean and clean != "null":
                                price = float(clean)
                                if price > 0:
                                    self.add_log(f"💰 Found valid price: {field} = €{price:.2f}")
                                    return f"€{price:.2f} ({label})"
                    except Exception as e:
                        self.add_log(f"⚠️  Error parsing {field}: {e}")
                        continue
            
            self.add_log("❌ No valid price fields found")
            return None
            
        except Exception as e:
            self.add_log(f"❌ Price extraction error: {e}")
            return None
    
    def show_captures(self):
        """Show list of captured cards"""
        if not self.captured_cards:
            self.add_log("📸 No cards captured yet")
            return
        
        self.add_log(f"\n📸 CAPTURED CARDS ({len(self.captured_cards)}):")
        for i, capture in enumerate(self.captured_cards[-5:], 1):
            self.add_log(f"  {i}. {capture['filename']}")
            self.add_log(f"     Confidence: {capture['confidence']:.3f}")
            # Show first line of OCR text
            ocr_text = capture['ocr_results'].get('best_text', 'No text')
            first_line = ocr_text.split('\n')[0] if ocr_text else 'No text'
            self.add_log(f"     Text: {first_line[:50]}")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_scanning = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state='normal')
        self.manual_scan_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
        
        self.status_label.config(text="⏹️ STOPPED")
        self.add_log("⏹️ Camera stopped")
        
        # Keep captured card displayed when camera stops
        if self.captured_card_image is None:
            self.detection_label.configure(image='', text="Camera stopped")
        else:
            self.detection_label.configure(image='', text="Camera stopped")
    
    def add_log(self, message):
        """Add log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.results_text.insert(tk.END, log_entry)
        self.results_text.see(tk.END)
    
    def close_app(self):
        """Close application"""
        self.is_scanning = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def run(self):
        """Start application"""
        try:
            self.root.mainloop()
        finally:
            self.close_app()