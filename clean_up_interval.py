import json
import argparse
import ffmpeg

def merge_clips(clips, threshold=0.5):
    """
    This method cleans up small gaps due to character motion or miniscule occlusions by merging videos with less than 0.5 gaps.
    """
    clips.sort(key=lambda x: x["start_time"])

    merged_clips = []
    current_clip = clips[0]

    for i in range(1, len(clips)):
        next_clip = clips[i]

        # Check if the next clip starts within the threshold
        if next_clip["start_time"] - current_clip["end_time"] <= threshold:
            # Merge by updating the end time and adding bounding boxes
            current_clip["end_time"] = max(current_clip["end_time"], next_clip["end_time"])
            current_clip["bounding_boxes"].extend(next_clip["bounding_boxes"])
        else:
            # Save the merged clip and move to the next one
            merged_clips.append(current_clip)
            current_clip = next_clip

    # Append the last processed clip
    merged_clips.append(current_clip)

    # Rename filenames sequentially
    for idx, clip in enumerate(merged_clips, start=1):
        clip["filename"] = f"clip_{idx}.mp4"

    return merged_clips


def main():
    parser = argparse.ArgumentParser(description="Clean up results")
    parser.add_argument("--video", type=str, help="Path to the input video file.", required=True)
    parser.add_argument("--json", type=str, help="Path to the json video", required=True)
    
    args = parser.parse_args()
    # This downloads the weights outside of the multiprocessing module. 
    metadata = None
    with open(args.json, 'r') as f:
        metadata = json.load(f)
    metadata = merge_clips(metadata)


    for i, clip in enumerate(metadata):
        
        output_filename = f"clip_{i+1}.mp4"
        try:
            ffmpeg.input(args.video, ss=clip["start_time"], to=clip["end_time"]).output(output_filename, c="copy").run()
        except:
            print("Video extraction error for clip. Video clip is most likely too small: ", output_filename)

    with open("metadata.json", 'w') as file:
        json.dump(metadata, file, indent=4)


    print("Face tracking and clip extraction completed!")


   

if __name__ == "__main__":
    main()
