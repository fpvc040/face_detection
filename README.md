# Face Detection and Video Clipping
This python script takes in a video and a reference image, and extracts (With audio) clips of the person detected. 
Tested on a Mac M1 using CPU

The script utilizes multi-processing to speed up the face detection algorith. 
The script also utilizes ffmpeg-python to extract video frames in a quick, consistent matter. The script also copies over the audio from the original clip.



 ## Requirements
The python script has 2 main dependencies: DeepFace and ffmpeg. The requirement files is attached to the document.

Recommended system: Linux/Mac (Unix)

If running on windows, the application requires an additional step to install ffmpeg executable. It can be downloaded from the following link. Make sure it is added to system path.
https://github.com/BtbN/FFmpeg-Builds/releases#:~:text=ffmpeg%2Dmaster%2Dlatest%2Dwin64%2Dgpl%2Dshared.zip

## Installation steps: 

If you want to use conda to set up an environment (Optional):
' conda create -n <name> '
' conda activate <name>'

Then, install the packages using:
'pip install -r requirements.txt'

That's it! You should be done with the install steps. 

If you don't want to use the requirements.txt file, you can install with the following line:
' pip install opencv-python ffmpeg-python torch tqdm deepface' 

## Usage:
' python script.py <video_path> <reference_image_path> [--interval N] ' 

Argument	Description
video_path	Path to the input video file.
reference_image_path	Path to the reference image of the face to track.
--interval N (optional)	Process every Nth frame (default: 1). Higher values improve speed but may reduce accuracy.

## Output
The outputs are in the result folder in this repo, but below is a drive link with the video as well. 





