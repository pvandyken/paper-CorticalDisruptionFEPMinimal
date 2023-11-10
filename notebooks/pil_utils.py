from PIL import Image

def vstack_images(images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    width = max(image.width for image in images)
    # Add up all the heights.
    height = sum(image.height for image in images)
    composite = Image.new('RGB', (width, height))
    # Paste each image below the one before it.
    y = 0
    for image in images:
        composite.paste(image, (0, y))
        y += image.height
    return composite

def hstack_images(images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    width = sum(image.width for image in images)
    # Add up all the heights.
    height = max(image.height for image in images)
    composite = Image.new('RGB', (width, height))
    # Paste each image below the one before it.
    x = 0
    for image in images:
        composite.paste(image, (x, 0))
        x += image.width
    return composite

def fig_to_pil(fig):
    # return img
    fig.canvas.draw()
    fig.canvas.get_renderer()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    fig.clf()
    return img