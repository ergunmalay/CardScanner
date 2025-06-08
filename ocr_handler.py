"""
OCR Handler Module for Card Scanner
Handles both Typhoon and Tesseract OCR engines
"""

import cv2
import os
import re

# Try importing OCR engines
try:
    from typhoon_ocr import ocr_document
    TYPHOON_AVAILABLE = True
except ImportError:
    TYPHOON_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class OCRHandler:
    def __init__(self, typhoon_available=None, tesseract_available=None):
        """Initialize OCR handler with available engines"""
        # Use passed values or check availability
        self.typhoon_available = typhoon_available if typhoon_available is not None else TYPHOON_AVAILABLE
        self.tesseract_available = tesseract_available if tesseract_available is not None else TESSERACT_AVAILABLE
        
        # Set default engine
        if self.typhoon_available:
            self.default_engine = "typhoon"
        elif self.tesseract_available:
            self.default_engine = "tesseract"
        else:
            self.default_engine = None
    
    def perform_ocr(self, card_image, engine_choice):
        """Perform OCR analysis with selected engine"""
        try:
            results = {
                'methods': {},
                'best_text': '',
                'full_text': '',
                'confidence': 0,
                'engine': 'none'
            }
            
            if engine_choice == "typhoon" and self.typhoon_available:
                # Use Typhoon OCR
                text_result = self.run_typhoon_ocr(card_image)
                results['methods']['typhoon'] = text_result
                results['best_text'] = text_result
                results['full_text'] = text_result
                results['confidence'] = 95
                results['engine'] = 'typhoon'
                
            elif engine_choice == "tesseract" and self.tesseract_available:
                # Use Tesseract OCR
                text_result = self.run_tesseract_ocr(card_image)
                results.update(text_result)
                results['engine'] = 'tesseract'
                
            else:
                # No OCR available
                results['methods']['error'] = "No OCR engine available"
                results['best_text'] = "No OCR engine available"
                results['full_text'] = "No OCR engine available"
                results['engine'] = 'error'
            
            return results
            
        except Exception as e:
            return {'error': str(e), 'best_text': '', 'full_text': '', 'engine': 'error'}
    
    def run_typhoon_ocr(self, card_image):
        """Run Typhoon OCR on card image and return FULL text"""
        try:
            # Save card image temporarily
            temp_filename = "temp_card_for_ocr.png"
            cv2.imwrite(temp_filename, card_image)
            
            try:
                markdown_result = ocr_document(temp_filename)
                
                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                # Extract text from result - RETURN EVERYTHING
                if isinstance(markdown_result, dict) and 'natural_text' in markdown_result:
                    full_text = markdown_result['natural_text']
                elif isinstance(markdown_result, str):
                    full_text = markdown_result
                else:
                    full_text = str(markdown_result)
                
                # Clean up the text but return everything
                cleaned_text = self.clean_full_ocr_text(full_text)
                return cleaned_text
                
            except Exception as e:
                return f"Typhoon OCR error: {e}"
                
        except Exception as e:
            return f"OCR setup error: {e}"
    
    def clean_full_ocr_text(self, full_text):
        """Clean OCR text but preserve all content"""
        try:
            # Remove markdown formatting but keep content
            cleaned = re.sub(r'[#*_`\[\]]', '', full_text)
            
            # Clean up excessive whitespace
            cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Remove empty lines
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)  # Trim lines
            
            # Remove common OCR artifacts
            cleaned = re.sub(r'\s*-\s*$', '', cleaned, flags=re.MULTILINE)  # Trailing dashes
            
            return cleaned.strip()
            
        except Exception as e:
            return full_text
    
    def run_tesseract_ocr(self, card_image):
        """Run Tesseract OCR and return full text"""
        try:
            results = {
                'methods': {},
                'best_text': '',
                'full_text': '',
                'confidence': 0
            }
            
            # Convert to grayscale and enhance
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Method 1: Full OCR
            text1 = pytesseract.image_to_string(enhanced, config='--psm 6').strip()
            results['methods']['full_ocr'] = text1
            
            # Method 2: With preprocessing
            thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            text2 = pytesseract.image_to_string(thresh, config='--psm 6').strip()
            results['methods']['preprocessed'] = text2
            
            # Choose best result
            best_text = text1 if len(text1) > len(text2) else text2
            
            results['best_text'] = best_text
            results['full_text'] = best_text
            results['confidence'] = 75
            
            return results
            
        except Exception as e:
            return {'methods': {'error': str(e)}, 'best_text': '', 'full_text': '', 'confidence': 0}