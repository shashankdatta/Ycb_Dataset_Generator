from PIL import Image

img = Image.open("test1.jpg")
img = img.resize((640, 640), Image.Resampling.LANCZOS)
img.save("../test1_resized.jpg", 'JPEG', quality=95)