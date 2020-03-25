from PIL import Image
import os

def convert_to_square(img):
    w,h = img.size
    max_side = max(w,h)
    img = img.crop((0,0,max_side,max_side))
    return img

if __name__ == '__main__':
    path = r'F:\YOLO_img'
    imgdir = os.listdir(path)
    for i,e in enumerate(imgdir):
        im = Image.open(os.path.join(path,e))
        im = convert_to_square(im)
        im = im.resize((416,416))
        im.save('img/{}.jpg'.format(i))