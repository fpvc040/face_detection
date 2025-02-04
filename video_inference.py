import cv2
import json
import ffmpeg
import torch
import multiprocessing
import argparse
import time
from tqdm import tqdm
from functools import partial
from deepface import DeepFace

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for inference.")

def get_video_metadata(video_path):
    """Extracts frame rate and total frame count from the video."""
    video = cv2.VideoCapture(video_path)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_rate, total_frames

def merge_tracking_results(thread_results):
    """Merges tracking results from multiple processes and groups detections into clips."""
    all_detections = [detection for thread in thread_results for detection in thread]
    all_detections.sort(key=lambda d: d['frame'])
    
    grouped, current_group = [], []
    last_frame = None
    
    for detection in all_detections:
        if last_frame is None or detection['frame'] == last_frame + 1:
            current_group.append(detection)
        else:
            if current_group:
                grouped.append(current_group)
            current_group = [detection]
        last_frame = detection['frame']
    
    if current_group:
        grouped.append(current_group)
    
    return grouped

def process_frames(frame_indices, video_path, reference_image_path, speed):
    """Processes a chunk of frames, detecting the target face using DeepFace."""
    frame_indices, task_id = frame_indices
    video = cv2.VideoCapture(video_path)
    results = []

    for frame_count in tqdm(frame_indices, desc=f"Task {task_id}", position=task_id):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = video.read()
        if not success:
            continue
        if speed:
            result = DeepFace.verify(frame, reference_image_path, enforce_detection=False)
        else: 
            result = DeepFace.verify(frame, reference_image_path, model_name='ArcFace', 
                                 detector_backend='retinaface', enforce_detection=False)
        
        if result["verified"]:
            target = result["facial_areas"]["img1"]
            x, y, w, h = int(target["x"]), int(target["y"]), int(target["w"]), int(target["h"])
            results.append({"frame": frame_count, "x": x, "y": y, "width": w, "height": h})

    video.release()
    return results

def merge_fractional_clips(clips, threshold=0.5):
    """
    Cleans up small gaps due to character motion or miniscule occlusions by merging videos with less than 0.5 seconds gaps.
    """
    clips.sort(key=lambda x: x["start_time"])

    merged_clips = []
    current_clip = clips[0]

    for i in range(1, len(clips)):
        next_clip = clips[i]

        if next_clip["start_time"] - current_clip["end_time"] <= threshold:
            current_clip["end_time"] = max(current_clip["end_time"], next_clip["end_time"])
            current_clip["bounding_boxes"].extend(next_clip["bounding_boxes"])
        else:
            merged_clips.append(current_clip)
            current_clip = next_clip

    merged_clips.append(current_clip)

    for idx, clip in enumerate(merged_clips, start=1):
        clip["filename"] = f"clip_{idx}.mp4"

    return merged_clips

def extract_clips(video_path, grouped_clips, frame_rate):
    """Extracts video clips where the target face is detected and saves metadata."""
    metadata = []
    for i, clip in enumerate(grouped_clips):
        start_time, end_time = clip[0]["frame"] / frame_rate, clip[-1]["frame"] / frame_rate
        output_filename = f"clip_{i+1}.mp4"
        metadata.append({
            "filename": output_filename,
            "start_time": start_time,
            "end_time": end_time,
            "bounding_boxes": clip
        })
    metadata = merge_fractional_clips(metadata)

    for video in metadata:
        try:
            ffmpeg.input(video_path, ss=video["start_time"], to=video["end_time"]).output(video["filename"], c="copy").run()
        except:
            print("Video extraction error for clip. Video clip is most likely too small: ", output_filename)

    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Face tracking and clip extraction completed!")

def main():
    parser = argparse.ArgumentParser(description="Face tracking in video using DeepFace.")
    parser.add_argument("--video", type=str, help="Path to the input video file.", required=True)
    parser.add_argument("--reference", type=str, help="Path to the reference face image.", required=True)
    parser.add_argument("--interval", type=int, default=1, help="Frame interval for processing (default: 1).")
    parser.add_argument("--speed", type=bool, default=False, help="Use a VGG model for better speed, instead of RetinaFace for higher accuracy.")
    parser.add_argument("--num_processes", type=int, default=4, help="Spin up a multi-process pipeline to improve speed of inference.")

    args = parser.parse_args()
    # This downloads the weights outside of the multiprocessing module. 
    if args.speed:
        _ = DeepFace.verify(args.reference, args.reference, enforce_detection=False)
    else: 
        _ = DeepFace.verify(args.reference, args.reference, model_name='ArcFace', 
                                 detector_backend='retinaface', enforce_detection=False)
        
    

    frame_rate, total_frames = get_video_metadata(args.video)
    frame_indices = list(range(0, total_frames, args.interval))
    num_workers = min(multiprocessing.cpu_count(), args.num_processes)
    chunk_size = len(frame_indices) // num_workers
    frame_chunks = [frame_indices[i:i + chunk_size] for i in range(0, len(frame_indices), chunk_size)]
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(partial(process_frames, video_path=args.video, reference_image_path=args.reference, speed=args.speed), 
                           zip(frame_chunks, range(num_workers)))
    
    grouped_clips = merge_tracking_results(results)
    print("Processing time:", time.time() - start_time, "seconds")
    
    extract_clips(args.video, grouped_clips, frame_rate)

if __name__ == "__main__":
    main()
