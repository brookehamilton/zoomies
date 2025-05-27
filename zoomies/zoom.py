"""Main function to create infinite zoom videos.

Based on the code from https://github.com/v8hid/infinite-zoom-stable-diffusion (MIT
license)
"""

from diffusers import StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from zoomies.utils import write_video, shrink_and_paste_on_blank
import numpy as np
import torch
import time

def zoom(
    prompts,
    negative_prompt='frames, borderline, text, garish, simple, empty, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur',
    model_id='stabilityai/stable-diffusion-2-inpainting',
    num_outpainting_steps=12,
    guidance_scale=7,
    num_inference_steps=50,
    custom_init_image=None
):
    """Create an infinite zoom video.

    'prompts' is a dictionary with keys as the frame number (integer) and values as the
    prompt for that frame.  e.g.: {0: 'bears', 12: 'squares', 24: 'chairs'}
    """

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to("cuda")

    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    g_cuda = torch.Generator(device='cuda')

    height = 512
    width = height

    current_image = Image.new(mode="RGBA", size=(height, width))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255-mask_image).convert("RGB")
    current_image = current_image.convert("RGB")
    if (custom_init_image):
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS)
    else:
        init_images = pipe(prompt=prompts[min(k for k in prompts.keys() if k >= 0)],
                           negative_prompt=negative_prompt,
                           image=current_image,
                           guidance_scale=guidance_scale,
                           height=height,
                           width=width,
                           mask_image=mask_image,
                           num_inference_steps=num_inference_steps)[0]
        current_image = init_images[0]
    mask_width = 128
    num_interpol_frames = 30

    all_frames = []
    all_frames.append(current_image)

    for i in range(num_outpainting_steps):
        print('Outpaint step: ' + str(i+1) +
              ' / ' + str(num_outpainting_steps))

        prev_image_fix = current_image

        prev_image = shrink_and_paste_on_blank(current_image, mask_width)

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        images = pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
                      negative_prompt=negative_prompt,
                      image=current_image,
                      guidance_scale=guidance_scale,
                      height=height,
                      width=width,
                      # generator = g_cuda.manual_seed(seed),
                      mask_image=mask_image,
                      num_inference_steps=num_inference_steps)[0]
        current_image = images[0]
        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image
            interpol_width = round(
                (1 - (1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2
            )
            interpol_image = interpol_image.crop((interpol_width,
                                                  interpol_width,
                                                  width - interpol_width,
                                                  height - interpol_width))

            interpol_image = interpol_image.resize((height, width))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (height-2*mask_width) / (height-2*interpol_width)) / 2*height
            )
            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2)
            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            all_frames.append(interpol_image)
        all_frames.append(current_image)
        interpol_image.show()

    video_file_name = "infinite_zoom_" + str(time.time())
    fps = 30
    save_path = video_file_name + ".mp4"
    start_frame_dupe_amount = 15
    last_frame_dupe_amount = 15

    write_video(save_path, all_frames, fps, True,
                start_frame_dupe_amount, last_frame_dupe_amount)
    return save_path
