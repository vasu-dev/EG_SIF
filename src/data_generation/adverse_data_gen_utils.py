import cv2
import random 
import numpy as np


def cutout(number, patch_size, prob, cutout_inside = True, patch_color = 255):
    '''
    Adds random small cutouts pathces at ramdom places in the image
    number : list of number of patches to add, chosen  randomly
    patch_size : patch size
    prob : Probability of the applying the cutout 
    cutout_inside : weather to start with an offset from image egdes
    patch_color : color of the patch
    '''
    num = random.choice(number)
    patch_size_half = patch_size // 2
    offset = 1 if patch_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > prob:
            return image
        
        else:
            h, w, _ = image.shape
            #h, w = image.shape
            if cutout_inside:
                cxmin, cxmax = patch_size_half, w + offset - patch_size_half
                cymin, cymax = patch_size_half, h + offset - patch_size_half
            else:
                cxmin, cxmax = 0, w + offset
                cymin, cymax = 0, h + offset
            for i in range(num):
                cx = np.random.randint(cxmin, cxmax)
                cy = np.random.randint(cymin, cymax)
                xmin = cx - patch_size_half
                ymin = cy - patch_size_half
                xmax = xmin + patch_size
                ymax = ymin + patch_size
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                image[ymin:ymax, xmin:xmax] = patch_color

            return image
    
    return _cutout

def patch(patch_color, offset, prob):
    
    '''
    Adds a big patch to a random loction along the edge of the image
    patch_color : color of the patch
    offset : a list of offset(distance from edge) to choose from  
    prob : Probability of the applying the patch noise
    '''

    def _patch(image):
        image = np.asarray(image).copy()

        if np.random.random() > prob:
            return image
        else:
            off = random.choice(offset)
            h, w, _ = image.shape
            #h, w = image.shape
            x = [0,w//4, w//3, 2*w//4, 2*w//3, 3*w//4]
            xmin = int(random.choice(x))
            y = [0, h-off]
            ymin = int(random.choice(y))
            image[ymin:ymin+off, xmin:xmin+off] = patch_color

            return image
        
    return _patch


def sp_noise(prob, noise_prob):
    '''
    Add salt and pepper noise to image
    prob : Probability of the applying the s&p noise
    noise_prob : Probability of the noise
    
    '''
    def _sp_noise(image):
        image = np.asarray(image).copy()
        
        if np.random.random() > prob:
            return image

        # Getting the dimensions of the image 
        row , col, _ = image.shape 
        total_px = row*col
        
        # Randomly pick some pixels in the image for coloring them white 
        # Pick a random number between 0 & total_px*noise_prob
        number_of_pixels = random.randint(0, int(total_px*noise_prob)) 
        for i in range(number_of_pixels): 
        
            # Pick a random y coordinate 
            y_coord=random.randint(0, row - 1) 
          
            # Pick a random x coordinate 
            x_coord=random.randint(0, col - 1) 
          
            # Color that pixel to white 
            image[y_coord][x_coord] = 255
          
        # Randomly pick some pixels in 
        # the image for coloring them black 
        # Pick a random number between 300 and 10000 
        number_of_pixels = random.randint(0 , int(total_px*noise_prob)) 
        for i in range(number_of_pixels): 
        
            # Pick a random y coordinate 
            y_coord=random.randint(0, row - 1) 
          
            # Pick a random x coordinate 
            x_coord=random.randint(0, col - 1) 
          
            # Color that pixel to black 
            image[y_coord][x_coord] = 0
          
        return image 
    
    return _sp_noise

def eyeglass(prob, thickness = 1):
    '''
    Adds a eyeglass like patch occluding the eye  
    prob: Probability of the noise
    '''
    transparency = random.randint(0,150)

    def _eyeglass(img):

        if np.random.random() > prob:
            return img
        else:
            line_start_x = random.randint(0,img.shape[0])
            line_start_y = 0
            clicked_x    = random.randint(img.shape[1]//3,2*img.shape[1]//3)
            clicked_y    = random.randint(img.shape[0]//3,2*img.shape[0]//3)
            line_end_y   = img.shape[1]
            line_end_x   = random.randint(3*img.shape[1]//4,img.shape[1])

            pts = np.array([ [line_start_x, line_start_y],
                            [clicked_x, clicked_y],
                            [line_end_x, line_end_y] ], np.int32)

            coeffs = np.polyfit(pts[:,1], pts[:,0], 2)
            poly   = np.poly1d(coeffs)

            yarr = np.arange(line_start_y, line_end_y)
            xarr = poly(yarr)
            parab_pts = np.array([xarr, yarr],dtype=np.int32).T
            cv2.polylines(img, [parab_pts], False, (transparency,0,0), thickness)

            return img
    
    return _eyeglass 