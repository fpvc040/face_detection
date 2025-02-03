# Face Detection and Video Clipping

This Python script processes a video and a reference image to extract clips (with audio) where the specified person is detected. 
It has been tested on a Mac M1 using CPU and utilizes multi-processing for faster face detection.

The script employs `ffmpeg-python` for efficient frame extraction while preserving audio from the original clip. It successfully extracts the clips with audio in seconds.

The script also uses `multi-processing` using Pool.map() to speed up the inference. I've included visual progress bars for each process to track progress. 
<img width="306" alt="image" src="https://github.com/user-attachments/assets/cdffe5ca-5b3a-4aa0-a544-813a5854dd4a" />


## Requirements

The script has two main dependencies: `DeepFace` and `ffmpeg`. The required dependencies are listed in `requirements.txt`.

### **Recommended System:**
- Linux/Mac (Unix)

#### **Windows Users:**
Windows (additional setup required for FFmpeg)

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
python video_inference.py --video <video_path> --reference <reference_image_path> [--interval N] [--speed] [--num_processes P]
```

### **Arguments**
| Argument | Description |
|----------|-------------|
| `--video` | Path to the input video file. (Required) |
| `--reference` | Path to the reference image of the face to track. (Required) |
| `--interval N` (optional) | Process every Nth frame (default: 1). Higher values improve speed but may reduce accuracy. |
| `--speed` (optional) | Use a VGG model for faster processing instead of RetinaFace (default) for higher accuracy. |
| `--num_processes P` (optional) | Number of processes to use for multiprocessing (default: 4). |

## Output
- Extracted video clips (`clip_1.mp4`, `clip_2.mp4`, etc.)
- `metadata.json` containing timestamps and bounding boxes of detected faces.

The outputs are stored in the `Results` folder. Additionally, a drive link with processed videos is available.
[Drive folder link with a preview in Home Directory](https://drive.google.com/drive/folders/1223QfyY7tHeG3OsyHDNQfKf4lDLKR7ib?usp=sharing)

## Assumptions:
- This assumes that ideally, each frame is analyzed. If the frames are skipped( Eg: Every 5 frames), then the video creation can be problematic. 
- The face should be visible enough in the video such that the features are visible in most of the scene. If the face features are miniscule, then the face embeddings will not be picked up. This should work well for mose video types (professional shoot or webcam)
- The model can be set up to run batch image inference using the model weights and pytorch/TF directly in addition;this project optimizes inference using multiprocessing. 


