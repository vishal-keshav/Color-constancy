import numpy as np
import cv2

def adjust_gamma(img, gamma = 1.0):
    inv_gamma = 1.0/gamma
    gamma_map = np.array([((i/255.0)**inv_gamma)*255
                    for i in np.arange(0,256)]).astype("uint8")
    corrected = cv2.LUT(img, gamma_map)
    return corrected

def show_image(img,img_title = "Image"):
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_mask(img, msk):
    #res = cv2.bitwise_and(img, img, mask = msk)
    idx=(msk==1)
    img[idx] = 0
    return img

def main():
    img = cv2.imread('IMG5.png')
    blackLevel = 10;
    (B, G, R) = cv2.split(img)
    saturationLevel = np.maximum(np.maximum(np.max(R), np.max(G)), np.max(B)) - 2
    print(saturationLevel)
    B = B - blackLevel
    G = G - blackLevel
    R = R - blackLevel
    R[R < 0] = 0
    G[G < 0] = 0
    B[B < 0] = 0
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = mask + (R>=saturationLevel-blackLevel).astype("uint8")
    mask = mask + (G>=saturationLevel-blackLevel).astype("uint8")
    mask = mask + (B>=saturationLevel-blackLevel).astype("uint8")
    mask = mask > 0
    mask[200:, 200:] = 1
    img = cv2.merge([B, G, R])
    show_image(img)
    img = apply_mask(img, mask)
    show_image(img)
    img = adjust_gamma(img, gamma = 2.2)
    show_image(img)

if __name__ == "__main__":
    main()
