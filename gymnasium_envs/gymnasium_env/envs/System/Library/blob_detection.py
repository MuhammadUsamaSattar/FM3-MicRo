import cv2


def iden_blob_detect(im, particle_locs, particle_rads):
    """Generates particle location and size

    Args:
        im : Image from which to extract the particle data
        particle_loc : List of size 2 containing previous location of particle
        particle_rad : Previous radius of particle

    Returns:
        bool: Determines if particle has been lost
        list : List of size 2 containing new location of particle
        int : New radius of particle
    """
    blob_lost = False

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 1
    # params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.minArea = 400
    params.maxArea = 10000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Creates a detector with the parameters depending upon which version of openCV the user has
    ver = (cv2.__version__).split(".")
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(im)  # Detects blob(s)

    # Assigns new particle location and radius if particle was detected.
    # Otherwise, return blob_lost as True.
    if len(keypoints) > 0:
        particle_locs = [list(map(int, keypoint.pt)) for keypoint in keypoints]
        particle_rads = [int(keypoint.size / 2) for keypoint in keypoints]

    else:
        blob_lost = True

    return blob_lost, [particle_locs[0]], [particle_rads[0]]
