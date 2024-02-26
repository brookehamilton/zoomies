"""
Utility functions

These were copied from the original notebook by unknown author.
"""

from PIL import Image
import imageio
import numpy as np

def image_grid(imgs, rows, cols):
    """
    Takes a list of images and makes them into a grid with the specified number of
    rows and columns
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def shrink_and_paste_on_blank(current_image, mask_width):
  """
  Decreases size of current_image by mask_width pixels from each side,
  then adds a mask_width width transparent frame,
  so that the image the function returns is the same size as the input.
  :param current_image: input image to transform
  :param mask_width: width in pixels to shrink from each side
  """

  height = current_image.height
  width = current_image.width

  #shrink down by mask_width
  prev_image = current_image.resize((height-2*mask_width,width-2*mask_width))
  prev_image = prev_image.convert("RGBA")
  prev_image = np.array(prev_image)

  #create blank non-transparent image
  blank_image = np.array(current_image.convert("RGBA"))*0
  blank_image[:,:,3] = 1

  #paste shrinked onto blank
  blank_image[mask_width:height-mask_width,mask_width:width-mask_width,:] = prev_image
  prev_image = Image.fromarray(blank_image)

  return prev_image

def load_img(address, res=(512, 512)):
    if address.startswith('http://') or address.startswith('https://'):
        image = Image.open(requests.get(address, stream=True).raw)
    else:
        image = Image.open(address)
    image = image.convert('RGB')
    image = image.resize(res, resample=Image.LANCZOS)
    return image

def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames = frames[::-1]

    # Get dimensions of the frames
    w, h = frames[0].size

    # Create an imageio video writer
    writer = imageio.get_writer(file_path, fps=fps)

    # Duplicate the start and end frames
    start_frames = [frames[0]] * start_frame_dupe_amount
    end_frames = [frames[-1]] * last_frame_dupe_amount

    # Write the duplicated frames to the video writer
    for frame in start_frames:
        # Convert PIL image to numpy array
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the frames to the video writer
    for frame in frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the duplicated frames to the video writer
    for frame in  end_frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Close the video writer
    writer.close()
