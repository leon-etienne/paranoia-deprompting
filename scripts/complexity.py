"""Repeat blur and sharpen filters to make a tiny image-complexity experiment."""

from PIL import ImageFilter
from diffusers.utils import export_to_video, load_image

# Change these values, then run the file.
image_url = "http://picsum.photos/seed/64/512"
iterations = 1000
blur_radius = 2
sharpen_radius = 4
sharpen_percent = 150
output = "out.mp4"
fps = 60

# Keep every intermediate image so the process can become a video.
image = load_image(image_url).convert("RGB")
frames = [image]

for i in range(iterations):
    # Blur removes small local details.
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Sharpen tries to recover edges, but it does not return the original image.
    image = image.filter(ImageFilter.UnsharpMask(sharpen_radius, sharpen_percent))

    frames.append(image)

export_to_video(frames, output, fps=fps)
print(output)
