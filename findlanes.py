import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

FILENAME = "solidYellowCurve.jpg"

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def slope(x1,y1,x2,y2):
    return ((y2-y1)/(x2-x1))

def line_y_int(m, x, y):
    # y = mx + b
    # return b = y - mx
    return y - m*x

        
def line_endpoints(img, slope, yint):
    imshape = img.shape

    y1 = imshape[0]
    x1 = int((y1 - yint)/slope)

    y2 = int(imshape[0]/2 + 60)
    x2 = int((y2 - yint)/slope)

    return x1,y1,x2,y2

def line_len(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_points = {'X' : [], 'Y' : []}
    right_points = {'X' : [], 'Y' : []}

    # left lane stats
    ll_avg_slope = 0
    ll_avg_x = 0
    ll_avg_y = 0
    # right lane stats
    rl_avg_slope = 0
    rl_avg_x = 0
    rl_avg_y = 0

    ishape = img.shape
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = slope(x1,y1,x2,y2) # as in y = mx + b
            if (math.isinf(m)):
                continue
            if m < -0.6 and m > -0.9:
                left_points['X'] += [x1,x2]
                left_points['Y'] += [y1,y2]
                ll_avg_slope += m
                ll_avg_x += x1 + x2
                ll_avg_y += y1 + y2
            if m > 0.5 and m < 0.9:
                right_points['X'] += [x1,x2]
                right_points['Y'] += [y1,y2]
                rl_avg_slope += m
                rl_avg_x += x1 + x2
                rl_avg_y += y1 + y2
    
    # draw left line
    if len(left_points['X']) > 0:
        ll_num_points = len(left_points['X'])
        ll_num_lines = int(ll_num_points / 2)
        # average slope, and average point on left line
        ll_avg_slope /= ll_num_lines
        ll_avg_x = int(ll_avg_x / ll_num_points)
        ll_avg_y = int(ll_avg_y / ll_num_points)
        # get y-intercept, and two points on line. draw it
        ll_y_int = line_y_int(ll_avg_slope, ll_avg_x, ll_avg_y)
        # print('# lines = ', ll_num_lines, ' and ll_avg_slope = ', ll_avg_slope)
        x1,y1,x2,y2 = line_endpoints(img, ll_avg_slope, ll_y_int)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    if len(right_points['X']) > 0:
        # draw right line
        rl_num_points = len(right_points['X'])
        #print(rl_num_points)
        rl_num_lines = math.ceil(rl_num_points / 2)
        # average slope, average point on right line
        rl_avg_slope /= rl_num_lines
        rl_avg_x = int(rl_avg_x / rl_num_points)
        rl_avg_y = int(rl_avg_y / rl_num_points)
        # get y-intercept, and two points on the line. draw it
        rl_y_int = line_y_int(rl_avg_slope, rl_avg_x, rl_avg_y)
        x1,y1,x2,y2 = line_endpoints(img, rl_avg_slope, rl_y_int)
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)


    # Python: cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) → None
    cv2.circle(img, (ll_avg_x, ll_avg_y), 20, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def get_verticies(image):
    imshape = image.shape

    x1,y1 = 125, imshape[0]
    x2,y2 = imshape[1]/2 - 25, imshape[0]/2 + 50
    x3,y3 = imshape[1]/2 + 25,imshape[0]/2 + 50
    x4,y4 = imshape[1] - 70,imshape[0]

    return np.array([[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]], dtype=np.int32)

def process_image(image):
    # convert to grayscale
    gray = grayscale(image)

    # apply gaussian blurring to the image
    kernel_size = 5
    blurred_gray = gaussian_blur(gray, kernel_size)

    # canny edge detection
    low=50
    high=150
    edges = cv2.Canny(blurred_gray, low, high)

    region_verticies = get_verticies(edges)
    region = region_of_interest(edges, region_verticies)

    # get the hough lines
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 100
    max_line_gap = 150
    lines = hough_lines(region, rho, theta, threshold, min_line_length, max_line_gap)

    lines_on_orig_image = cv2.addWeighted(lines, 1, image, 1, 0)

    return lines_on_orig_image

def process_image_from_path(img_path):
    return process_image(mpimg.imread(img_path))

'''
image_paths = os.listdir('test_images/')
for img_path in image_paths:
    print('Processing ' + 'test_images/' + img_path)
    pimg = process_image_from_path('test_images/' + img_path)
    mpimg.imsave('test_images/' + 'lines_' + img_path, pimg)
'''

print('processing video')
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile('test_videos/lanes_solidWhiteRight.mp4')