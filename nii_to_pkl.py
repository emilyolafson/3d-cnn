# Read nii.gz files and save to picked format.

import os
import nibabel as nib

data_dir =  #/path/to/nii/scans

# get a list of subjects by listing all files in a given directory (data_dir)
# check that the order of subjects in scan_paths matches the order of subjects in your y_data
scan_paths = [
    os.path.join(os.getcwd(), data_dir, x)
    for x in os.listdir(data_dir)]

scans = np.array([process_scan(path) for path in scan_paths])
all_x=scans

#load y data (scores to predict). needs to be in the same order as the files in data_dir. usually this will be ordered numerically lowest-highest.
all_y = pd.read_excel("/path/to/y/data/data.xlsx", header=None)
all_y = all_y.to_numpy()


pickle_out= open(data_dir + "/all_xdata.pkl", "wb")
pickle.dump(all_x, pickle_out)
pickle_out.close()

pickle_out= open(data_dir + "/all_ydata.pkl", "wb") 
pickle.dump(all_y, pickle_out)
pickle_out.close()


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired dimensions for input to 3D CNN.
    desired_depth = 64
    desired_width = 64 
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def normalize(volume):
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

