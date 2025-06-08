"""
Utility Functions for Card Scanner
"""

import cv2
import numpy as np
from difflib import SequenceMatcher
import re

def calculate_similarity(text1, text2):
    """Calculate string similarity between two texts"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def create_price_overlay_image(base_image, price_info):
    """Create image with price overlay if price exists"""
    try:
        # Create a copy of the base image
        overlay_image = base_image.copy()
        
        if price_info and price_info.get('price') != "No Price Found":
            height, width = overlay_image.shape[:2]
            
            # Create semi-transparent overlay at bottom
            overlay_height = min(80, height // 3)
            overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
            overlay[:] = (0, 50, 0)  # Dark green background
            
            # Add price text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Card name (top line)
            name_text = f"FOUND: {price_info['name'][:20]}"
            cv2.putText(overlay, name_text, (5, 20), font, 0.4, (255, 255, 255), 1)
            
            # Price (main line)
            price_text = f"PRICE: {price_info['price']}"
            cv2.putText(overlay, price_text, (5, 45), font, 0.5, (0, 255, 0), 2)
            
            # Confidence score
            conf_text = f"Match: {price_info['confidence']:.1%}"
            cv2.putText(overlay, conf_text, (5, 65), font, 0.35, (200, 200, 200), 1)
            
            # Combine with original image
            if height >= overlay_height:
                # Place overlay at bottom
                start_y = height - overlay_height
                overlay_image[start_y:, :] = overlay
            else:
                # If image too small, blend overlay
                overlay_resized = cv2.resize(overlay, (width, height))
                overlay_image = cv2.addWeighted(overlay_image, 0.3, overlay_resized, 0.7, 0)
        
        return overlay_image
        
    except Exception as e:
        print(f"❌ Price overlay error: {e}")
        return base_image

def extract_card_name_from_text(full_text):
    """Extract likely card name from OCR text"""
    try:
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        # Skip description-like lines
        skip_words = ['action', 'attack', 'defense', 'once per turn', 'when', 'if', 'target', 
                      'deal', 'prevent', 'destroy', 'goes again', 'cost', 'combat chain']
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip lines with game mechanics
            if any(skip in line_lower for skip in skip_words):
                continue
            
            # Look for title-like patterns (short, no game terms)
            if len(line.split()) <= 6 and len(line) >= 3:
                # Clean the line
                cleaned = re.sub(r'[^\w\s]', ' ', line)
                cleaned = ' '.join(cleaned.split())
                
                if cleaned:
                    return cleaned
        
        # Fallback: return first non-empty line (first 4 words)
        if lines:
            first_line = re.sub(r'[^\w\s]', ' ', lines[0])
            first_line = ' '.join(first_line.split()[:4])
            return first_line
        
        return ""
        
    except Exception as e:
        return ""

def clean_ocr_text(text):
    """Clean up OCR text output"""
    try:
        # Remove markdown formatting
        cleaned = re.sub(r'[#*_`\[\]]', '', text)
        
        # Clean up whitespace
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned, flags=re.MULTILINE)
        
        # Remove artifacts
        cleaned = re.sub(r'\s*-\s*$', '', cleaned, flags=re.MULTILINE)
        
        return cleaned.strip()
        
    except Exception as e:
        return text