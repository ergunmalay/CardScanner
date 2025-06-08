#!/usr/bin/env python3
"""
YOLO CardMarket Scanner - Main Entry Point
"""

import os
import sys

# Check dependencies before importing
def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    dependencies_ok = True
    
    # Core dependencies
    try:
        import cv2
        print("✅ OpenCV (cv2) OK")
    except ImportError:
        print("❌ Missing: pip install opencv-python")
        dependencies_ok = False
    
    try:
        from PIL import Image, ImageTk
        print("✅ Pillow (PIL) OK")
    except ImportError:
        print("❌ Missing: pip install Pillow")
        dependencies_ok = False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLO) OK")
    except ImportError:
        print("❌ Missing: pip install ultralytics")
        dependencies_ok = False
    
    try:
        import tkinter
        print("✅ Tkinter OK")
    except ImportError:
        print("❌ Tkinter not installed (usually comes with Python)")
        dependencies_ok = False
    
    # OCR engines
    typhoon_available = False
    tesseract_available = False
    
    try:
        from typhoon_ocr import ocr_document
        typhoon_available = True
        print("✅ Typhoon OCR available")
    except ImportError:
        print("⚠️  Typhoon OCR not available (optional: pip install typhoon-ocr)")
    
    try:
        import pytesseract
        tesseract_available = True
        print("✅ Tesseract OCR available")
    except ImportError:
        print("⚠️  Tesseract not available (optional: pip install pytesseract)")
    
    if not typhoon_available and not tesseract_available:
        print("❌ No OCR engines available! Install at least one:")
        print("   pip install typhoon-ocr")
        print("   pip install pytesseract")
        dependencies_ok = False
    
    return dependencies_ok, typhoon_available, tesseract_available

def main():
    """Main function"""
    print("🚀 YOLO CardMarket Scanner")
    print("=" * 50)
    print("📄 Version: 2.0 (Modular)")
    print("🔄 Features:")
    print("  - YOLO-based card detection")
    print("  - Full OCR text extraction")
    print("  - Merged card/price database")
    print("  - Persistent price display")
    print("  - 4-panel visualization")
    print("=" * 50)
    
    # Check dependencies first
    deps_ok, typhoon_ok, tesseract_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first")
        input("Press Enter to exit...")
        return 1
    
    print("\n✅ All required dependencies OK")
    
    # Check for merged data file
    if not os.path.exists("cards_with_prices.json"):
        print("\n⚠️  WARNING: cards_with_prices.json not found!")
        print("💡 Run 'python merge_json_files.py' first to create it")
        print("   The scanner will still start but won't be able to lookup prices\n")
    
    # Import scanner after dependency check
    try:
        print("Loading scanner module...")
        from scanner import YOLOCardScanner
        
        # Pass OCR availability to scanner
        print("Creating scanner instance...")
        scanner = YOLOCardScanner(typhoon_ok, tesseract_ok)
        
        print("Starting GUI...")
        scanner.run()
        
    except ImportError as e:
        print(f"❌ Error importing scanner module: {e}")
        print("Make sure scanner.py is in the same directory")
        input("Press Enter to exit...")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())