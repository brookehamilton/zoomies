import imageio
import numpy as np
import os
import requests
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from PIL import Image


def outpaint(
    image: Image.Image,
    api_key: str,
    prompt: str,
    prop: float = 1.0,
    output_format: str = "png",
    style_preset: str = "fantasy-art",
    creativity: float = 1.0,
) -> Image.Image:
    """
    Outpaint an image using the Stability AI API.
    """

    # Infer the dimensions to add based on the aspect ratio
    add_lr = int(image.width * prop / 2)
    add_tb = int((image.height / image.width) * add_lr)

    # Prepare the image for the API request
    image_buffer = BytesIO()
    image.save(image_buffer, format=output_format)
    image_buffer.seek(0)

    # Make the API request
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/outpaint",
        headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
        files={"image": image_buffer},
        data={
            "left": add_lr,
            "right": add_lr,
            "up": add_tb,
            "down": add_tb,
            "prompt": prompt,
            "output_format": output_format,
            "style_preset": style_preset,
            "creativity": creativity,
        },
    )

    if response.status_code == 200:
        # Load the response content into an Image object
        outpainted_image = Image.open(BytesIO(response.content))
        return outpainted_image
    else:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )


def sequential_outpaint(
    image_start: Image.Image,
    api_key: str,
    prompts: dict[str, str],
    prop: float = 1.0,
    output_format: str = "png",
    style_preset: str = "fantasy-art",
    creativity: float = 1.0,
    verbose: bool = True,
    force_remake: bool = False,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    Create a sequence of outpainted images and interpolate between them.

    `prompts` should be a dict where keys are filenames and values are prompts.

    Starting image should be the *innermost* one in the infinite zoom, and
    prompts should go from inner to outer.
    """

    # Initialize the current image as the starting image
    image_current = image_start

    # Loop through each outpainting prompt
    out_fullsize = []
    out_downsize = [image_start]
    for filename, prompt in prompts.items():
        # If force_remake is False, check if the outpainted image already exists
        if not force_remake:
            try:
                # Attempt to load the outpainted image from the file system
                image_out = Image.open(f"{filename}").convert("RGB")
                if verbose:
                    print(f"Loaded existing outpainted image for prompt: {prompt}")
            except FileNotFoundError:
                # If the file does not exist, proceed with outpainting
                image_out = None
                if verbose:
                    print(
                        f"{filename} not found, subsequent images will be regenerated."
                    )
                force_remake = True
        else:
            image_out = None

        # If the outpainted image does not exist or we are forcing remakes,
        # perform outpainting and save
        if image_out is None:
            if verbose:
                print(f"Outpainting with prompt: {prompt}")

            # Outpaint the current image
            image_out = outpaint(
                image=image_current,
                api_key=api_key,
                prompt=prompt,
                prop=prop,
                output_format=output_format,
                style_preset=style_preset,
                creativity=creativity,
            )

            # Save the outpainted image
            if verbose:
                print(f"Saving outpainted image to {filename}")
            image_out.save(f"{filename}", format=output_format)

        # Append the interpolated images to the output stack
        out_fullsize.append(image_out)

        # Update the current image to the downscaled version of the outpainted
        # image for the next iteration
        image_current = image_out.resize(
            (image_start.width, image_start.height), Image.Resampling.LANCZOS
        )
        out_downsize.append(image_current)

    # Reverse orders to have innermost image last
    out_fullsize.reverse()
    out_downsize.reverse()

    return out_fullsize, out_downsize


def interpolate(
    image_start: Image.Image, image_end: Image.Image, steps: int = 30
) -> list[Image.Image]:
    """
    Interpolate between two images.

    Dimensions of output will be the same as the end image.  Start image is
    assumed to have the same aspect ratio as the end image but larger
    (typically the output of the outpaint function).
    """

    # Use a geometric progression to scale the start image down to the end image
    #
    # Here calculating how much we'd crop each iteration if we could take
    # fractions of pixels using the geometric progression
    img_size_ratio = image_end.width / image_start.width
    scales = [img_size_ratio ** (i / steps) for i in range(steps + 1)]
    crops_floating = [
        (image_start.width * (1 - s) / 2, image_start.height * (1 - s) / 2)
        for s in scales
    ]

    # To calculate integer crops, use an "error-diffusion (Bresenham style)"
    # algorithm to ensure that "jumps" happen in the most natural spots rather
    # than due to arbitrary rounding stuff
    prev_x = prev_y = 0
    crops_integer = []
    for crop_x_f, crop_y_f in crops_floating:
        # How far from the previous crop to the floating-point ideal?
        diff_x = crop_x_f - prev_x
        diff_y = crop_y_f - prev_y

        # Select integer crops to get closest to the floating-point ideal
        step_x = round(diff_x)
        step_y = round(diff_y)
        crop_x_i = prev_x + step_x
        crop_y_i = prev_y + step_y

        # Store the integer crops
        prev_x = crop_x_i
        prev_y = crop_y_i
        crops_integer.append((crop_x_i, crop_y_i))

    # Loop through each step and create interpolated images
    out_stack = []
    for i in range(steps + 1):
        # Number of pixels to crop from each side
        crop_width, crop_height = crops_integer[i]

        # Crop the start image
        image_cropped = image_start.crop(
            (
                crop_width,
                crop_height,
                image_start.width - crop_width,
                image_start.height - crop_height,
            )
        )

        # Resize to match the end image size
        image_resized = image_cropped.resize(
            (image_end.width, image_end.height), Image.Resampling.LANCZOS
        )
        out_stack.append(image_resized)

    return out_stack


def sequential_interpolate(
    stack_fullsize: list[Image.Image],
    stack_downsize: list[Image.Image],
    steps: int = 30,
    max_workers: int | None = None,
) -> list[Image.Image]:
    """
    Create a sequence of interpolated images between each pair of images in the stack.
    """

    assert len(stack_fullsize) == len(stack_downsize) - 1

    # Determine number of workers
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # Parallelize the interpolation process
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = list(
            exe.map(
                interpolate,
                stack_fullsize,
                stack_downsize[1:],
                [steps] * len(stack_fullsize),
            )
        )

    # Remove first frame from all but the first interpolated stack
    results = [results[0]] + [r[1:] for r in results[1:]]

    # Flatten the list of lists into a single list
    out_stack = sum(results, [])

    return out_stack


def stack_to_video(
    stack: list[Image.Image],
    file_path: str,
    fps: int = 30,
    start_frame_dupe_amount: int = 15,
    last_frame_dupe_amount: int = 30,
) -> None:
    """
    Save a list of images as a video file.
    """
    # Duplicate the first and last frames
    stack = (
        stack[:1] * start_frame_dupe_amount
        + stack
        + stack[-1:] * last_frame_dupe_amount
    )

    # Convert images to numpy arrays
    frames = [np.asarray(image) for image in stack]

    # Write the video file
    #
    # Using ChatGPT's suggestions for optimal output to YouTube 1080p
    # fmt: off
    ffmpeg_params=[
        "-preset", "slow",
        "-crf", "18",
        "-profile:v", "high",
        "-level", "4.2",
        "-movflags", "+faststart",
    ]
    # fmt: on
    writer = imageio.get_writer(
        file_path,
        format="FFMPEG",
        macro_block_size=1,
        codec="libx264",
        fps=fps,
        ffmpeg_params=ffmpeg_params,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


if __name__ == "__main__":
    # Read API key from bjk_stability_key
    with open("bjk_stability_key", "r") as f:
        stability_key = f.read().strip()

    # Create a sequence of outpainted images and interpolate between them
    prompts = {
        "01-stormy-ocean.png": "Pirates sailing through a stormy ocean, dark clouds, high waves, dramatic lighting",
        "02-octopus-mushrooms.png": "A giant octopus with mushrooms growing on its tentacles, underwater scene, bioluminescent lighting",
        "03-sunken-temple.png": "Old gods, a sunken temple, ancient ruins, coral reefs, mysterious atmosphere",
        "04-space-casino.png": "A futuristic space casino, neon lights, floating platforms, alien architecture",
        "05-ogre-1.png": "Inside the jaws of a giant ogre, dark and eerie atmosphere, glowing eyes",
        "06-ogre-2.png": "Inside the jaws of a giant ogre, dark and eerie atmosphere, glowing eyes, more detailed",
        "07-ogre-3.png": "The face of a giant ogre, dark and eerie atmosphere, glowing eyes, highly detailed",
        "08-aztec-god.png": "An Aztec god, surrounded by ancient artifacts, vibrant colors, mystical energy",
        "09-aztec-god-2.png": "An Aztec god, surrounded by ancient artifacts, vibrant colors, blood sacrifice",
        "10-frame.png": "A beautiful wood frame surrounds the dark and eerie scene, ornate carvings, intricate details",
        "11-frame.png": "A classic wood frame surrounds the painting of the Aztec god, ornate carvings, intricate details",
        "12-frame.png": "An ornate wooden painting frame, hanging the Aztec god painting, intricate carvings, golden accents",
        "13-museum.png": "A bright white museum gallery with the Aztec god painting on display, elegant lighting, marble floors, people admiring the art",
        "14-museum-2.png": "A bright white museum gallery with the Aztec god painting on display, elegant lighting, marble floors, people admiring the art",
        "15-museum-3.png": "A bright white museum gallery with the Aztec god painting on display, elegant lighting, marble floors, people admiring the art",
        "16-museum-4.png": "Daytime, bright white light, sunshine, sunlight, windows",
        "17-waterfall-1.png": "The waterfall scene, bright and sunny, vibrant colors, lush greenery",
        "18-waterfall-2.png": "The waterfall scene, bright and sunny, vibrant colors, lush greenery, sunshine",
        "19-waterfall-3.png": "The waterfall scene, bright and sunny, vibrant colors, lush greenery, sunshine, swimmers",
        "20-gemstone-1.png": "Close detail of a large gemstone, sparkling, vibrant colors, intricate facets",
        "21-gemstone-2.png": "Close detail of a large gemstone, sparkling, vibrant colors, intricate facets, rainbow, sparkling",
        "22-gemstone-3.png": "Kaleidoscopic view of a large gemstone, sparkling, vibrant colors, intricate facets, rainbow, sparkling",
        "23-unicorn-rainbow-1.png": "Majestic unicorns soar through the rainbow sky, vibrant colors, magical atmosphere",
        "24-unicorn-rainbow-2.png": "Majestic unicorns soar through the rainbow sky, vibrant colors, mushrooms, sky squid",
        "25-unicorn-rainbow-3.png": "Majestic unicorns soar through the rainbow sky, vibrant colors, mushrooms, sky squid, bright colors",
        "26-snowglobe-1.png": "A magical snow globe with a unicorn and rainbow inside, snow outside, blizzard, whiteout",
        "27-snowglobe-2.png": "A magical snow globe with a unicorn and rainbow inside, snow outside, blizzard, whiteout, snowflakes",
        "28-snowglobe-3.png": "A magical snow globe sits on a wooden table outside a cottage, snow outside, blizzard, whiteout, snowflakes",
        "29-snowglobe-4.png": "A magical snow globe sits on a wooden table outside a cottage, snow outside, blizzard, whiteout, snowflakes, warm light",
        "30-snowglobe-5.png": "A magical snow globe sits on a wooden table outside a cottage, snow outside, blizzard, whiteout, snowflakes, warm light, cozy atmosphere",
        "31-ice-cream-1.png": "We see the very top of a giant ice cream cone, with a cherry on top, colorful sprinkles, and a bright blue sky in the background",
        "32-ice-cream-2.png": "We see the top of a giant ice cream cone, with a cherry on top, colorful sprinkles, and a bright blue sky in the background, waffle cone",
        "33-ice-cream-3.png": "What looks like snow on the ground is actually a giant ice cream cone, with a cherry on top, colorful sprinkles, and a bright blue sky in the background, waffle cone",
        "34-ice-cream-4.png": "What looks like snow on the ground is actually a giant ice cream cone, with a cherry on top, colorful sprinkles, waffle cone, summer day, bright blue sky",
        "35-ice-cream-5.png": "What looks like snow on the ground is actually a giant ice cream cone, with a cherry on top, colorful sprinkles, waffle cone, summer day, bright blue sky",
        "36-ice-cream-6.png": "What looks like snow on the ground is actually a giant ice cream cone, with a cherry on top, colorful sprinkles, waffle cone, summer day, bright blue sky, people enjoying ice cream",
        "37-ice-cream-7.png": "What looks like snow on the ground is actually a giant ice cream cone, with a cherry on top, colorful sprinkles, waffle cone, summer day, bright blue sky, people enjoying ice cream",
    }
    stack_full, stack_down = sequential_outpaint(
        image_start=Image.open("lighthouse.jpg").convert("RGB"),
        api_key=stability_key,
        prompts=prompts,
        prop=1.0,
        output_format="png",
        style_preset="fantasy-art",
        creativity=0.9,
    )
    stack_interpolated = sequential_interpolate(
        stack_fullsize=stack_full,
        stack_downsize=stack_down,
        steps=30,
        max_workers=20,
    )
    stack_to_video(
        stack=stack_interpolated,
        file_path="zoomies.mp4",
        fps=30,
        start_frame_dupe_amount=30,
        last_frame_dupe_amount=30,
    )
