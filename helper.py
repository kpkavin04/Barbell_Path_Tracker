import webcolors

def color_name_to_bgr(name):
    try:
        rgb = webcolors.name_to_rgb(name.lower())
        return (rgb.blue, rgb.green, rgb.red)  # Convert RGB to BGR
    except ValueError:
        return (128, 0, 0)  # Default to navy if invalid color
