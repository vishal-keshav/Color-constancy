import numpy as np
import cv2

def adjust_gamma(img, gamma = 1.0):
    inv_gamma = 1.0/gamma
    gamma_map = np.array([((i/255.0)**inv_gamma)*255
                    for i in np.range(0,256)]).astype("uint8")
    corrected = cv2.LUT(img, gamma_map)

def show_image(img,img_title = "Image"):
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_mask(img, msk):
    res = cv2.bitwise_and(img, img, mask = msk)
    return res

def main():
    img = cv2.imread('1056.png')
    blackLevel = 2048;
    (B, G, R) = cv2.split(img)
    saturationLevel = np.maximum(np.maximum(R, G), B) - 2
    B = B - blackLevel
    G = G - blackLevel
    R = R - blackLevel
    R[R < 0] = 0
	G[G < 0] = 0
	B[B < 0] = 0
    mask = np.zeros(img.shape(0), img.shape(1));
    mask += mask+double(R>=saturationLevel-blackLevel)
    mask += mask+double(G>=saturationLevel-blackLevel)
    mask += mask+double(B>=saturationLevel-blackLevel)
    mask[1050:, 2050:] = 1
    img = apply_mask(cv2.merge([B, G, R]), mask)
    img = adjust_gamma(img, gamma = 2.2)
    show(img)

if __name__ == "__main__":
    main()
