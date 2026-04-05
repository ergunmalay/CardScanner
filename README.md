# CardScanner-Retry

A real-time trading card scanner built using Python, OpenCV, and Tesseract OCR.

## Overview

CardScanner-Retry is designed to detect, extract, and identify trading cards from a live camera feed. It focuses on accuracy, speed, and usability, making it suitable for applications such as pack openings, live streams, and inventory tracking.

The system processes frames in real time, isolates card regions, performs OCR on key areas, and retrieves relevant data such as card names and pricing.

## Features

- Real-time card detection using contour filtering
- Perspective correction and card warping
- OCR extraction using Tesseract (optimized via ctypes wrapper)
- Targeted text recognition (card titles, key fields)
- Price lookup integration (Cardmarket data)
- Multi-panel debugging interface for visibility into pipeline stages
- Designed for OBS integration during live recordings

## Tech Stack

- Python
- OpenCV
- Tesseract OCR (C API via ctypes)
- JSON data processing
- Optional: MMDetection (for deep learning-based detection)

## Pipeline

1. Capture frame from webcam
2. Detect potential card contours
3. Warp detected card to a normalized view
4. Extract relevant regions (title, identifiers)
5. Perform OCR on selected regions
6. Match results against dataset
7. Fetch and display price data

## Use Case

- Trading card pack openings
- Live streaming overlays
- Inventory management tools
- Card price tracking systems

## Status

Private project. Actively developed and iterated.

## Notes

This project explores the intersection of computer vision and real-time data processing, with a focus on practical usability rather than just model accuracy.
