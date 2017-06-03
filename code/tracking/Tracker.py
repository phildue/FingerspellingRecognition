import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
#from tracker.MeanShiftSegmentor import mean_shift_segmentation


def main():
    # Load an color image in colour
    dir_rsrc = '../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/A/a/'
    test_a = 'color_0_0002.png'
    img = cv2.imread(dir_rsrc + dir_rsrc + dir_a + test_a, 1)
    #img_show = Image.new("RGBA", img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 16)
    draw.text((0, 0), "This is a test", (255, 55, 100), font)
    draw = ImageDraw.Draw(img)
    cv2.imshow('image', img)
    cv2.waitKey(500)
    #img = mean_shift_segmentation(img)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


main()