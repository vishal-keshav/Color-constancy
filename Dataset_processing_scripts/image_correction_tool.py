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
"""
Pass None as argumets if not known
"""
def create_mask(img, init_mask, blackLevel, saturationLevel):
    # Images are expected to be parsed in BGR, default by cv2
    (B, G, R) = cv2.split(img)
    if saturationLevel == None:
        saturationLevel =np.maximum(np.maximum(np.max(R), np.max(G)),
                            np.max(B)) - 2
    if blackLevel == None:
        blackLevel = 10 # Assuming 8 bit depth per channel
    B = B - blackLevel
    G = G - blackLevel
    R = R - blackLevel
    # Remove negatives
    R[R < 0] = 0
    G[G < 0] = 0
    B[B < 0] = 0
    if init_mask.all() == None:
        init_mask = np.zeros((img.shape[0], img.shape[1]))
    init_mask = init_mask + (R>=saturationLevel-blackLevel).astype("uint8")
    init_mask = init_mask + (G>=saturationLevel-blackLevel).astype("uint8")
    init_mask = init_mask + (B>=saturationLevel-blackLevel).astype("uint8")
    init_mask = init_mask > 0
    img = cv2.merge([B, G, R])
    return img, init_mask

def correct_image(img, mask_info):
    if mask_info!=None:
        #Logic to create mask from mask info here
        mask = None
        pass
    else: # Remove else part
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask[200:, 200:] = 1
    img, mask = create_mask(img, mask, 10, None)
    img = apply_mask(img, mask)
    img = adjust_gamma(img, 2.2)
    return img

def correct_image_list(img_list, mask_info_list):
    assert len(img_list) == len(mask_info_list)
    for i in range(len(img_list)):
        img_list[i] = correct_image(img_list[i], mask_info_list[i])
    return img_list

def main():
    img = cv2.imread('IMG5.png')
    img = correct_image(img, None)
    show_image(img)

if __name__ == "__main__":
    main()
