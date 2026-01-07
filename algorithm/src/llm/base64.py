import base64, io
import numpy as np
from PIL import Image

def np_to_data_url(img: np.ndarray, fmt="PNG") -> str:
    """
    img: HxWxC (uint8) or HxW (uint8). If float, please convert to uint8 first.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def image_to_data_url(image, fmt="PNG") -> str:
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        raise ValueError("Unsupported image string; expected data URL.")

    if isinstance(image, np.ndarray):
        return np_to_data_url(image, fmt=fmt)

    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"

    raise TypeError(f"Unsupported image type: {type(image)}")

# if __name__ == "__main__":
