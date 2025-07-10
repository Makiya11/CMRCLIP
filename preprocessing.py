import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from typing import List, Tuple, Optional
from utils import clean_text
from sklearn.model_selection import train_test_split

# Configuration constants
ROOT_PATH = '.'
DATA_PATH = f'{ROOT_PATH}/data/CMR_cine_lge'
CSV_PATH = f'{ROOT_PATH}/img_text/text2npy.csv'

# Image and video specifications
IMAGE_SIZE = (224, 224)
CINE_FRAME_COUNT = 30
LGE_SA_MIN_DEPTH = 10
LGE_SA_MAX_DEPTH = 15
LGE_OTHER_DEPTH = 1
VIDEO_FPS = 1

# View types
CINE_VIEWS = ['cine_lax', 'cine_sa']
LGE_VIEWS = ['lge_sa', 'lge_lax', 'lge_2ch', 'lge_3ch']
ALL_VIEWS = CINE_VIEWS + LGE_VIEWS


def validate_image_quality(row: pd.Series, views: List[str]) -> bool:
    """
    Validate image quality based on shape requirements for different view types.
    
    Args:
        row: DataFrame row containing file paths
        views: List of view types to validate
        
    Returns:
        bool: True if all views meet quality requirements, False otherwise
    """
    try:
        quality_checks = []
        
        for view in views:
            if pd.isna(row[view]):
                quality_checks.append(False)
                continue
                
            try:
                shape = np.load(row[view]).shape
            except Exception:
                quality_checks.append(False)
                continue
            
            # Check basic image dimensions
            if shape[0] != IMAGE_SIZE[0] or shape[1] != IMAGE_SIZE[1]:
                quality_checks.append(False)
                continue
            
            # View-specific quality checks
            if view in CINE_VIEWS:
                # Cine views should have 30 frames
                quality_checks.append(shape[3] == CINE_FRAME_COUNT)
            elif view == 'lge_sa':
                # LGE short axis should have 10-15 slices
                quality_checks.append(LGE_SA_MIN_DEPTH <= shape[2] <= LGE_SA_MAX_DEPTH)
            elif view in LGE_VIEWS:
                # Other LGE views should have 1 slice
                quality_checks.append(shape[2] == LGE_OTHER_DEPTH)
            else:
                print(f'Unknown view type: {view}')
                quality_checks.append(False)
        
        return all(quality_checks)
        
    except Exception as e:
        print(f"Error validating image quality: {e}")
        return False


def compute_total_video_length(row: pd.Series, views: List[str]) -> int:
    """
    Compute total video length across all views.
    
    Args:
        row: DataFrame row containing file paths
        views: List of view types to process
        
    Returns:
        int: Total video length in frames
    """
    total_length = 0
    
    for view in views:
        try:
            shape = np.load(row[view]).shape
            
            if view in CINE_VIEWS:
                # Cine views: depth * time
                total_length += shape[2] * shape[3]
            elif view == 'lge_sa':
                # LGE short axis: just depth
                total_length += shape[2]
            elif view in LGE_VIEWS:
                # Other LGE views: single frame
                total_length += shape[2]
                
        except Exception as e:
            print(f"Error processing view {view}: {e}")
            
    return total_length


def create_video_from_arrays(row: pd.Series, views: List[str], output_path: str) -> None:
    """
    Convert numpy arrays to MP4 video file.
    
    Args:
        row: DataFrame row containing file paths
        views: List of view types to process
        output_path: Path to save the output video
    """
    video_frames = []
    lge_sa_length = 1  # Default length
    
    # Get LGE SA length for synchronization
    if 'lge_sa' in views:
        try:
            lge_sa_length = np.load(row['lge_sa']).shape[2]
        except Exception:
            print("Warning: Could not load LGE SA data")
    
    # Process each view
    for view in views:
        try:
            img_array = np.load(row[view])
            
            if view in CINE_VIEWS:
                # Extract middle slice from cine views
                width, height, depth, time = img_array.shape
                mid_slice_idx = int(np.floor(depth / 2))
                mid_slice_cine = img_array[:, :, mid_slice_idx, :].squeeze()
                video_frames.append(mid_slice_cine)
                
            elif view == 'lge_sa':
                # Add LGE SA frames directly
                video_frames.append(img_array)
                
            elif view in LGE_VIEWS:
                # Repeat LGE frames to match SA length
                repeated_frames = [img_array] * lge_sa_length
                video_frames.extend(repeated_frames)
                
        except Exception as e:
            print(f"Error processing view {view}: {e}")
    
    if not video_frames:
        print("No valid frames to create video")
        return
    
    # Concatenate all frames
    try:
        video = np.concatenate(video_frames, axis=2)
        width, height, length = video.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (width, height))
        
        # Write frames
        for i in range(length):
            frame = cv2.cvtColor(video[:, :, i], cv2.COLOR_GRAY2BGR)
            out.write(frame)
        
        out.release()
        
    except Exception as e:
        print(f"Error creating video: {e}")


def setup_directories(data_path: str) -> Tuple[str, str]:
    """
    Create necessary directories for output.
    
    Args:
        data_path: Base data directory path
        
    Returns:
        Tuple of (videos_path, annotation_path)
    """
    videos_path = os.path.join(data_path, 'videos/all')
    annotation_path = os.path.join(data_path, 'annotation')
    
    for path in [data_path, videos_path, annotation_path]:
        os.makedirs(path, exist_ok=True)
    
    return videos_path, annotation_path


def preprocess_dataframe(df: pd.DataFrame, views: List[str]) -> pd.DataFrame:
    """
    Clean and preprocess the dataframe.
    
    Args:
        df: Input dataframe
        views: List of view types
        
    Returns:
        Cleaned dataframe
    """
    # Replace '0' with None for missing values
    df[views] = df[views].replace('0', None)
    
    # Update file paths
    for view in views:
        df[view] = df[view].str.replace('_20230306', '_2023')
    
    # Remove rows with missing data
    df = df.dropna(subset=views + ['impressions'])
    
    # Set up caption column
    df['caption'] = df['impressions']
    
    # Select relevant columns
    df_cmr = df[['caption', 'AccessionNumber', 'K_PAT_PROC_IMG_KEY'] + views].copy()
    
    # Filter by image quality
    df_cmr = df_cmr[df_cmr.apply(lambda x: validate_image_quality(x, views), axis=1)]
    
    # Calculate caption length
    df_cmr['caption_length'] = df_cmr['caption'].str.split().apply(len)
    
    # Reset index and create image IDs
    df_cmr = df_cmr.reset_index(drop=True)
    df_cmr['image_id'] = df_cmr.index.map(lambda x: f'video{x}')
    
    # Calculate video lengths
    df_cmr['video_length'] = df_cmr.apply(
        lambda x: compute_total_video_length(x, views), axis=1
    )
    
    return df_cmr


def main():
    """Main processing function."""
    print("Starting CMR data preprocessing...")
    
    # Load and sample data
    df = pd.read_csv(CSV_PATH)
    
    # Preprocess dataframe
    df_cmr = preprocess_dataframe(df, ALL_VIEWS)
    
    # Set up output directories
    videos_path, annotation_path = setup_directories(DATA_PATH)
    
    print(f"Processing {len(df_cmr)} samples...")
    
    # Generate videos
    for idx, row in df_cmr.iterrows():
        video_path = os.path.join(videos_path, f'{row["image_id"]}.mp4')
        create_video_from_arrays(row, ALL_VIEWS, video_path)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df_cmr)} videos")
    
    # Save annotations
    output_json = os.path.join(annotation_path, 'CMR.json')
    df_cmr.to_json(output_json, orient='records')
    
    print(f"Processing complete! Output saved to: {DATA_PATH}")
    print(f"Generated {len(df_cmr)} videos and annotations")


if __name__ == "__main__":
    main() 