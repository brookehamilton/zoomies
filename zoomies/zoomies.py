import imageio
import numpy as np
import requests
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


def interpolate(
    image_start: Image.Image, image_end: Image.Image, steps: int = 30
) -> list[Image.Image]:
    """
    Interpolate between two images.

    Dimensions of output will be the same as the end image.  Start image is
    assumed to have the same aspect ratio as the end image but larger
    (typically the output of the outpaint function).
    """

    # Calculate ratio of dimensions
    dim_ratio = image_end.width / image_start.width
    scale_base = dim_ratio ** (1 / steps)

    # Loop through each step and create interpolated images
    out_stack = []
    for i in range(steps + 1):
        # Calculate the interpolation factor
        scale = scale_base ** i

        # Number of pixels to crop from each side
        crop_width = int(image_start.width * (1 - scale) / 2)
        crop_height = int(image_start.height * (1 - scale) / 2)

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


def sequential_outpaint_and_interpolate(
    image_start: Image.Image,
    api_key: str,
    prompts: list[str],
    prop: float = 1.0,
    output_format: str = "png",
    style_preset: str = "fantasy-art",
    creativity: float = 1.0,
    interpolation_steps: int = 30,
    verbose: bool = True,
) -> list[Image.Image]:
    """
    Create a sequence of outpainted images and interpolate between them.

    Starting image should be the *innermost* one in the infinite zoom, and
    prompts should go from inner to outer.
    """

    # Initialize the current image as the starting image
    image_current = image_start

    # Loop through each outpainting prompt
    out = []
    for prompt in prompts:
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

        # Interpolate between the current and outpainted image
        interpolated_stack = interpolate(
            image_start=image_out, image_end=image_current, steps=interpolation_steps
        )

        # Append the interpolated images to the output stack
        out = interpolated_stack + out

        # Update the current image to the downscaled version of the outpainted
        # image for the next iteration
        image_current = interpolated_stack[0]

    return out


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
        stack[:1] * start_frame_dupe_amount + stack + stack[-1:] * last_frame_dupe_amount
    )

    # Convert images to numpy arrays
    frames = [np.asarray(image) for image in stack]

    # Write the video file
    imageio.mimwrite(file_path, frames, fps=fps, codec="libx264", quality=8)


if __name__ == "__main__":
    # Read API key from bjk_stability_key
    with open("bjk_stability_key", "r") as f:
        stability_key = f.read().strip()

    # Create a sequence of outpainted images and interpolate between them
    prompts = [
        "A serene landscape with a calm lake and mountains in the background",
        "A bustling cityscape with skyscrapers and busy streets",
        "A futuristic scene with flying cars and neon lights",
        "An alien world with strange plants and creatures",
        "A fantasy realm with castles and mythical beings",
    ]
    stack = sequential_outpaint_and_interpolate(
        image_start=Image.open("lighthouse.jpg").convert("RGB"),
        api_key=stability_key,
        prompts=prompts,
        prop=1.0,
        output_format="png",
        style_preset="fantasy-art",
        creativity=1.0,
        interpolation_steps=30
    )

    # Save the interpolated images as a video
    stack_to_video(
        stack=stack,
        file_path="outpainted_interpolation.mp4",
        fps=30,
        start_frame_dupe_amount=15,
        last_frame_dupe_amount=30,
    )
