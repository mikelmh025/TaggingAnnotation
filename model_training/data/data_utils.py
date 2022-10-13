import os
import torch

MODEL_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG','png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in MODEL_EXTENSIONS)

# Only one level of file
def make_image_set(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # remove hidden files
            if fname.startswith('.'):
                continue
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
        break
    return images

def get_mean_std_all(data_loader):
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for i, (img, label,idx) in enumerate(data_loader):
        psum    += img.sum(axis        = [0, 2, 3]) / (img.shape[2]*img.shape[3])
        psum_sq += (img ** 2).sum(axis = [0, 2, 3]) / (img.shape[2]*img.shape[3])
    count = len(data_loader) #* 256 * 256
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)
    return total_mean, total_std