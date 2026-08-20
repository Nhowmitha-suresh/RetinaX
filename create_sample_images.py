import os
from PIL import Image, ImageDraw

os.makedirs('sampleimages', exist_ok=True)
os.makedirs('frontend/web/assets/images', exist_ok=True)

def draw_fundus(filename, bg_color=(180, 50, 20), disc_color=(255, 230, 150), vessels=True):
    img = Image.new('RGB', (400, 400), (10, 10, 15))
    draw = ImageDraw.Draw(img)
    # Retina background
    draw.ellipse([20, 20, 380, 380], fill=bg_color)
    # Optic disc
    draw.ellipse([260, 160, 330, 230], fill=disc_color)
    # Macula
    draw.ellipse([140, 185, 175, 215], fill=(130, 30, 10))
    # Blood vessels
    if vessels:
        for offset in [-40, -20, 0, 20, 40]:
            draw.arc([100, 100 + offset, 300, 300 + offset], start=30, end=150, fill=(110, 20, 10), width=4)
            draw.arc([120, 80 + offset, 280, 320 + offset], start=200, end=320, fill=(100, 15, 10), width=3)
    img.save(filename)

draw_fundus('sampleimages/eye1.jpg', (190, 55, 20))
draw_fundus('sampleimages/eye2.png', (170, 40, 15))
draw_fundus('sampleimages/eye3.jpg', (200, 70, 30))
draw_fundus('sampleimages/eye4.jpg', (160, 35, 10))
draw_fundus('sampleimages/eye5.png', (185, 60, 25))

# Clinical team hero image generator
hero_img = Image.new('RGB', (600, 450), (15, 85, 80))
h_draw = ImageDraw.Draw(hero_img)
h_draw.rectangle([50, 50, 550, 400], fill=(240, 248, 245), outline=(20, 180, 160), width=4)
h_draw.text((120, 200), "RetinaX Clinical AI Team", fill=(10, 80, 75))
hero_img.save('frontend/web/assets/images/hero-banner.png')

print("Sample images generated successfully.")
