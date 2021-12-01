import os
import cv2
import numpy as np
import math
import dlib
import glob
from imutils import face_utils
from tqdm import tqdm
import time

def get_images_with_landmarks(image_folder, model_file = 'shape_predictor_68_face_landmarks.dat', verbose = 0):
    
    # Load dlib face detection and prediction
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_file)

    images = []
    landmarks = []

    if verbose >= 1: print(f'Step 1: Get {len(glob.glob(os.path.join(image_folder, "*.jpg")))} Images and Detect Faces')

    for f in tqdm(glob.glob(os.path.join(image_folder, "*.jpg")), disable = verbose<=0):
        
        
        if verbose >= 2: print("Processing file: {}".format(f))

        #img = dlib.load_rgb_image(f)
        img = cv2.imread(f)

        dets = detector(img, 1)

        #TODO pick largest face from dets instead of just first one
        
        try: 
            face_box = dets[0]
            if verbose >= 2: print("Found {} faces".format(len(dets)))
        except: 
            if verbose >= 2: print('Found no faces')
            continue # If dets doesn't have a zero-index (is 0 long), no faces found. skip iteration and go to next image

        # Get the landmarks/parts for the face
        shape = predictor(img, face_box)

        shape_as_np = face_utils.shape_to_np(shape)
        if verbose >= 2: print(shape_as_np)

        at = shape_as_np.T 
        face_shape_final = list(zip(at[0],at[1]))

        landmarks.append(face_shape_final)

        # Now load image
        img = cv2.imread(f)

            # Convert to floating point
        img = np.float32(img)/255.0

        images.append(img)
    
    return images, landmarks

def get_images_with_landmarks_and_style_on_em(image_folder, model_file = 'shape_predictor_68_face_landmarks.dat', verbose = 0):
    '''
    Same as above, but visualizes progress in a window that looks cool and scientific.
    '''

    # Load dlib face detection and prediction
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_file)
    win = dlib.image_window()

    images = []
    landmarks = []

    if verbose >= 1: print(f'Step 1: Getting {len(glob.glob(os.path.join(image_folder, "*.jpg")))} images')

    counter = 0
    for f in tqdm(glob.glob(os.path.join(image_folder, "*.jpg")), disable = verbose<=0):
        
        
        if verbose >= 2: print("Processing file: {}".format(f))

        img = dlib.load_rgb_image(f)
        
        if counter % 2 == 0:
            win.clear_overlay()
            win.set_image(img)
            time.sleep(1)

        dets = detector(img, 1)

        #TODO pick largest face from dets instead of just first one
        
        try: 
            face_box = dets[0]
            if verbose >= 2: print("Found {} faces".format(len(dets)))
        except: 
            if verbose >= 2: print('Found no faces')
            continue # If dets doesn't have a zero-index (is 0 long), no faces found. skip iteration and go to next image

        # Get the landmarks/parts for the face
        shape = predictor(img, face_box)
        
        if counter % 2 == 0: 
            win.add_overlay(shape)
            time.sleep(1)

        shape_as_np = face_utils.shape_to_np(shape)
        if verbose >= 2: print(shape_as_np)

        at = shape_as_np.T 
        face_shape_final = list(zip(at[0],at[1]))

        landmarks.append(face_shape_final)

        # Now load image
        img = cv2.imread(f)

            # Convert to floating point
        img = np.float32(img)/255.0

        images.append(img)
    
        counter += 1

    return images, landmarks

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)  
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([int(xin), int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([int(xout), int(yout)])
    
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    
    return tform[0]

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)
   
    # Insert points into subdiv
    for p in points:
            subdiv.insert((p[0], p[1]))

   
    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        

    
    return delaunayTri

def constrainPoint(p, w, h) :
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def black_to_white(img, threshold = 0.2):
    '''
    This creates spooky ghost images where the eyes are cut out and the outline of the face is weird.
    Use at your own risk.
    '''

    black_pixels = np.where(
    (img[:, :, 0] <= threshold) & 
    (img[:, :, 1] <= threshold) & 
    (img[:, :, 2] <= threshold)
    )

    img[black_pixels] = [255, 255, 255]

    return img

def crop_out_black(img, out_dim = 900, threshold = 0.2):
    '''
    Takes in an image of any dimension, crops out space that's all black (or close),
    resizes to a square of side length out_dim pixels.
    '''

    # This gives all the locations of nonblack pixels as two arrays
    non_black_pixels = np.where(
    (img[:, :, 0] >= threshold) & 
    (img[:, :, 1] >= threshold) & 
    (img[:, :, 2] >= threshold)
    )

    
    # Seeing in which dimension the nonblack pixels are wider
    # Ideally it's close

    nonblack_h_size = non_black_pixels[0].max() - non_black_pixels[0].min()
    nonblack_w_size = non_black_pixels[1].max() - non_black_pixels[1].min()

    if nonblack_h_size >= nonblack_w_size: # Image more high than wide

        upper_bound = non_black_pixels[0].max()
        lower_bound = non_black_pixels[0].min()

        horizontal_center = non_black_pixels[1].min() + nonblack_w_size/2
        left_bound = int(horizontal_center - nonblack_h_size/2)
        right_bound = int(horizontal_center + nonblack_h_size/2)
    
    else: 

        left_bound = non_black_pixels[1].min()
        right_bound = non_black_pixels[1].max()

        vertical_center = non_black_pixels[0].min() + nonblack_h_size/2
        upper_bound = int(vertical_center - nonblack_w_size/2)
        lower_bound = int(vertical_center + nonblack_w_size/2)
    
    img_cropped = img[lower_bound:upper_bound,left_bound:right_bound]

    img_resized = cv2.resize(img_cropped,(out_dim,out_dim))

    return img_resized

def average_image(image_path,
                  output_dim = 900,
                  upscale = 1,
                  verbose = 1,
                  **kwargs):
    '''
    Takes in the path to an image folder and dimensions of the output image. Returns an average of all the faces 
    it identifies in the images.
    :param image_path: string, The path to the image folder.
    :param output_dim: int, The x- and y-dimension of the output (output is always square)
    :param upscale: float, increases the resolution of the output, greatly hurts performance
    :param verbose: [0,1,2] 0 for no console output, 1 for some, 2 for far too much.
    '''
    
    disable_tqdm = verbose <= 0

    path = image_path
    
    # Dimensions of output image (scale back down later)
    w,h = int(output_dim*upscale), int(output_dim*upscale)
    
    if 'style_on_em' in kwargs:
        if kwargs['style_on_em']: images, allPoints = get_images_with_landmarks_and_style_on_em(path, verbose = 1)
        else: images, allPoints = get_images_with_landmarks(path, verbose=1)
    else: images, allPoints = get_images_with_landmarks(path, verbose=1)

    # Eye corners
    eyecornerDst = [ (int(0.3 * w ), int(h/3)), (int(0.7 * w ), int(h/3)) ]
    
    imagesNorm = []
    pointsNorm = []
    
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
    
    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32())
    
    n = len(allPoints[0])

    numImages = len(images)
    
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    
    if verbose >= 1: print('Step 2: Find Average Face Shape')
    for i in tqdm(range(0, numImages), disable = disable_tqdm):

        points1 = allPoints[i]

        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] 
        
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2))
        
        points = cv2.transform(points2, tform)
        
        points = np.float32(np.reshape(points, (68, 2)))
        
        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)
        
        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages
        
        pointsNorm.append(points)
        imagesNorm.append(img)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    
    # This next step fails sometimes, don't have a 100% fix
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))
    

    # Output image
    output = np.zeros((h,w,3), np.float32())

    if verbose >= 1: print('Step 3: Transform Faces to Average Shape')
    # Warp input images to average image landmarks
    for i in tqdm(range(0, len(imagesNorm)), disable = disable_tqdm) :
        img = np.zeros((h,w,3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)) :
            tin = []
            tout = []
            
            for k in range(0, 3) :                
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)
                
                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)
                
                tin.append(pIn)
                tout.append(pOut)
            
            
            warpTriangle(imagesNorm[i], img, tin, tout)


        # Add image intensities for averaging
        output = output + img


    # Divide by numImages to get average
    output = output / numImages

    # Crop to square without black and 900x900
    output = crop_out_black(output, out_dim=output_dim)

    return output