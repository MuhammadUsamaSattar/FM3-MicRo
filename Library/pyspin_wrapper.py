"""
Note:
This code was written was Jiangkun. Any questions should be directed to him.
"""

import PySpin


def Cam_PySpin_Init():
    """Initializes the PySpin camera allowing user to then read images from it

    Returns:
        bool: Determines if intialization was successful
        int: Number of detected PySpin cameras
        cam_list: List containing PySpin camera objects
        system: PySpin.System.GetInstance() which is used to get PySpin camera objects
    """
    system = PySpin.System.GetInstance()

    # # Get current library version
    # version = system.GetLibraryVersion()
    # print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print("Number of cameras detected: %d" % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        Cam_PySpin_Close(cam_list, system)
        print("No cameras!")
        return False, 0, 0

    return True, num_cameras, cam_list, system


def Cam_PySpin_Connect(cam):
    """Connects to the camera

    Args:
        cam : PySpin camera object

    Returns:
        bool: Determines if connection was successful
    """    
    nodemap_tldevice = cam.GetTLDeviceNodeMap()

    # Initialize camera
    cam.Init()

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    sNodemap = cam.GetTLStreamNodeMap()

    # Change bufferhandling mode to NewestOnly
    node_bufferhandling_mode = PySpin.CEnumerationPtr(
        sNodemap.GetNode("StreamBufferHandlingMode")
    )
    if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(
        node_bufferhandling_mode
    ):
        print("Unable to set stream buffer handling mode.. Aborting...")
        return False

    # Retrieve entry node from enumeration node
    node_newestonly = node_bufferhandling_mode.GetEntryByName("NewestOnly")
    if not PySpin.IsReadable(node_newestonly):
        print("Unable to set stream buffer handling mode.. Aborting...")
        return False

    # Retrieve integer value from entry node
    node_newestonly_mode = node_newestonly.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    # print('*** IMAGE ACQUISITION ***\n')

    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))

    if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(
        node_acquisition_mode
    ):
        print(
            "Unable to set acquisition mode to continuous (enum retrieval). Aborting..."
        )
        return False

    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
        "Continuous"
    )
    if not PySpin.IsReadable(node_acquisition_mode_continuous):
        print(
            "Unable to set acquisition mode to continuous (entry retrieval). Aborting..."
        )
        return False

    # Retrieve integer value from entry node
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
    # print('Acquisition mode set to continuous...')

    #  Begin acquiring images
    cam.BeginAcquisition()
    # print('Acquiring images...')

    #  Retrieve device serial number for filename
    device_serial_number = ""
    node_device_serial_number = PySpin.CStringPtr(
        nodemap_tldevice.GetNode("DeviceSerialNumber")
    )
    if PySpin.IsReadable(node_device_serial_number):
        device_serial_number = node_device_serial_number.GetValue()
        print("Device serial number retrieved as %s..." % device_serial_number)

    # new_gain = 6.0  # set new Gain for Cam
    # cam.Gain.SetValue(new_gain)
    current_gain = cam.Gain.GetValue()
    print("Device Gain is %s..." % str(current_gain))


def Cam_PySpin_GetImg(cam):
    """Retrieve next received image

    *** NOTES ***
    
    Capturing an image houses images on the camera buffer. Trying
    to capture an image that does not exist will hang the camera.
    
    Once an image from the buffer is saved and/or no longer
    needed, the image must be released in order to keep the
    buffer from filling up.

    Args:
        cam : PySpin camera object

    Returns:
        numpy array: Image data
    """    

    image_result = cam.GetNextImage(200)  # originally 1000

    #  Ensure image completion
    if image_result.IsIncomplete():
        print(
            "Image incomplete with image status %d ..." % image_result.GetImageStatus()
        )

    else:

        # Getting the image data as a numpy array
        image_data = image_result.GetNDArray()

    #  Release image
    #
    #  *** NOTES ***
    #  Images retrieved directly from the camera (i.e. non-converted
    #  images) need to be released in order to keep from filling the
    #  buffer.
    image_result.Release()

    return image_data


def Cam_PySpin_Stop(cam):
    """End acquisition
    
    *** NOTES ***
    Ending acquisition appropriately helps ensure that devices clean up
    properly and do not need to be power-cycled to maintain integrity.

    Args:
        cam : PySpin camera object
    """    
    cam.EndAcquisition()

    # Deinitialize camera
    cam.DeInit()

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam


def Cam_PySpin_Close(cam_list, system):
    """Closes the PySpin camera

    Args:
        cam_list : List containing PySpin camera objects
        system : System object
    """    
    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    # system.ReleaseInstance()  #(?)I don't know why it can not work, but it seems nothing happend if system is not released
