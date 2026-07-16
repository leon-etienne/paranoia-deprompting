from PIL import Image

from clip_token_lab.images import make_grid


def test_make_grid_dimensions():
    images = [Image.new("RGB", (10, 20)) for _ in range(3)]
    assert make_grid(images, columns=2).size == (20, 40)
