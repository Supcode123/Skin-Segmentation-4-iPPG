import os
import cv2
import numpy as np

from code_projects.tools.video_ppgi.roi_extraction import overlay


def roi_video_output(cropped_resized_frame, rois, fs,
                     output_dir, projects, key, kismed=False, task=None):
    overlayed_frames = []
    for i in range(len(cropped_resized_frame)):
        overlaid = overlay(cropped_resized_frame[i].astype(np.uint8), rois[i], (0, 255, 0), 0.3)
        overlayed_frames.append(overlaid)

    # Create a video writer
    if kismed:
        pass
    else:
        output_path = os.path.join(output_dir, projects[key] + '_' + 'pred_overlay.avi')
    h, w, _ = overlayed_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以换成 'mp4v' 输出 .mp4
    out = cv2.VideoWriter(output_path, fourcc, fs, (w, h))

    # Write all frames
    for frame in overlayed_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Release resources
    out.release()
    print(f"Video saved to {output_path}")



