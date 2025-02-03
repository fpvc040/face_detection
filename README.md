# Face Detection and Video Clipping

This Python script processes a video and a reference image to extract clips (with audio) where the specified person is detected. 
It has been tested on a Mac M1 using CPU and utilizes multi-processing for faster face detection.

The script employs `ffmpeg-python` for efficient frame extraction while preserving audio from the original clip.

## Requirements

The script has two main dependencies: `DeepFace` and `ffmpeg`. The required dependencies are listed in `requirements.txt`.

### **Recommended System:**
- Linux/Mac (Unix)
- Windows (additional setup required for FFmpeg)

#### **Windows Users:**
If running on Windows, you must install the FFmpeg executable separately. Download it from the link below and ensure it is added to your system PATH:
[FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases#:~:text=ffmpeg%2Dmaster%2Dlatest%2Dwin64%2Dgpl%2Dshared.zip)

## Installation

### **Optional: Set Up a Conda Environment**
```sh
conda create -n <env_name>
conda activate <env_name>
```

### **Install Dependencies**
Using `requirements.txt`:
```sh
pip install -r requirements.txt
```

Without `requirements.txt`:
```sh
pip install opencv-python ffmpeg-python torch tqdm deepface
```

## Usage
```sh
python script.py <video_path> <reference_image_path> [--interval N]
```

### **Arguments**
| Argument | Description |
|----------|-------------|
| `video_path` | Path to the input video file. |
| `reference_image_path` | Path to the reference image of the face to track. |
| `--interval N` (optional) | Process every Nth frame (default: 1). Higher values improve speed but may reduce accuracy. |

## Output
- Extracted video clips (`clip_1.mp4`, `clip_2.mp4`, etc.)
- `metadata.json` containing timestamps and bounding boxes of detected faces.

The outputs are stored in the `results` folder. Additionally, a drive link with processed videos is available.
