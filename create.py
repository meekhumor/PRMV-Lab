python3 -c "
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a sample text image for OCR testing
img = Image.new('RGB', (800, 400), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 36)
    font_small = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 24)
except:
    font = ImageFont.load_default()
    font_small = font

draw.text((40, 40),  'Hello, OCR World!', fill=(0,0,0), font=font)
draw.text((40, 100), 'PRMV Lab - Image to Text', fill=(30,30,180), font=font)
draw.text((40, 160), 'OpenCV + Tesseract OCR Demo', fill=(0,0,0), font=font_small)
draw.text((40, 200), 'Phone: +91 9876543210', fill=(0,0,0), font=font_small)
draw.text((40, 240), 'Email: lab@university.edu', fill=(0,0,0), font=font_small)
draw.text((40, 280), 'Date: 12 October 2026', fill=(0,0,0), font=font_small)
draw.text((40, 320), 'Recognition Accuracy: 98.5%', fill=(180,30,30), font=font_small)
img.save('/home/meekhumor/PRMV Lab/images/sample_text.png')
print('Sample image created!')
" 2>&1