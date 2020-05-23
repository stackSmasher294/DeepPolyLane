import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import numpy as np
import matplotlib.image as mpimg
import cv2
import functools as ft
import warnings

warnings.filterwarnings('ignore') # TODO: Not a good idea

class TUSimpleLaneDataset(Dataset):
    """ Lane detection dataset """
    
    def __init__(self, json_dataset_list, root_dir, transform=None, cache_enabled=False, cache_size=-1, degree=3):
        """
        Args:
            json_dataset_list: list of the data points described as JSON strings returned by file.readlines()
        """
        self.json_dataset_list = json_dataset_list
        self.root_dir = root_dir
        self.transform = transform
        self.degree = degree
        self.image_cache = {}
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size # -1: No limit on the cache size
        
    def __len__(self):
        return len(self.json_dataset_list)
    
    def __getitem__(self, idx):
        image_path, lanes_polyline = self.parse_dataset(self.json_dataset_list[idx], self.root_dir)
        if self.cache_enabled:
            if(not idx in self.image_cache):
                image = mpimg.imread(image_path)
                if self.cache_size == -1 or len(self.image_cache) <= self.cache_size:
                    self.image_cache[idx] = image
            else:
                image = self.image_cache[idx]
        else:
            image = mpimg.imread(image_path)
            
        height = np.shape(image)[0]
        width = np.shape(image)[1]
        
        lane_detections = self.get_polynomials_and_bounds(lanes_polyline, width, height)
        
        sample = {'image':image, 'detections':lane_detections}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    """
        Parses a new-line terminated json string and extracts the image name and the lane poly-lines from it.
    """
    def parse_dataset(self, json_string, path_prefix = ''):
        json_dict = json.loads(json_string)
        lanes = json_dict['lanes']
        num_lanes = len(lanes)
        lanes_polyline = [[] for _ in range(num_lanes)]
        h_samples = json_dict['h_samples']
        for h_idx, h in enumerate(h_samples):
            for lane_idx in range(num_lanes):
                w = lanes[lane_idx][h_idx]
                if w >= 0:
                    lanes_polyline[lane_idx].append((h, w))
        
        # Only include if non-empty
        lanes_polyline = [line for line in lanes_polyline if len(line) != 0]

        image_path = path_prefix + json_dict['raw_file']

        return (image_path, lanes_polyline)
    
    def get_polynomials_and_bounds(self, lanelines, img_width, img_height):
        num_lanes = len(lanelines)
        
        # columns: 4 for coefficients of the cubic polynomial f(y) and 2 for y limits
        # number of rows: 5, one for each lane line
        polynomials_and_bounds = np.zeros((6, 5), dtype=np.double) 
        
        for i, line in enumerate(lanelines):
            """
            h is the height pixel, and is chosen as the independent variable because of the way
            the lane lines curve in the image space.
            
            Also, rescale each point to its width or height so that it stays in the range [0,1]
            """
            x = [w / img_width for (h,w) in line]
            y = [h / img_height for (h,w) in line]
            
            coeffs = np.polyfit(y, x, self.degree)
            coeffs_and_bounds = np.append(coeffs, [min(y), max(y)]).T
            polynomials_and_bounds[:, i] = coeffs_and_bounds
            #polynomials_and_bounds[0, i] = 1.0 # because it exists
        return polynomials_and_bounds 
    

# tranforms
class Normalize(object):
    """Normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, detections = sample['image'], sample['detections']
        
        image_copy = np.copy(image)
        detections_copy = np.copy(detections)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0

        return {'image': image_copy, 'detections': detections_copy}
    
class NormalizeImageNet(object):
    def __init__(self):
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def __call__(self, sample):
        image_tensor, detections = sample['image'], sample['detections']
        output_tensor = self.normalize(image_tensor)
        return {'image': output_tensor, 'detections' : detections}

 
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, detections = sample['image'], sample['detections']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h_int, new_w_int = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w_int, new_h_int))

        return {'image': img, 'detections': detections}
    

    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self):
        pass

    def __call__(self, sample):
        image, detections = sample['image'], sample['detections']
         
        # if image has no grayscale color channel, add one
        #if(len(image.shape) == 2):
        #    # add that third color dim
        #    image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #detections_flattened = detections.reshape((1, -1))
        
        return {'image': torch.from_numpy(image),
                'detections': torch.from_numpy(detections)}
    
class ReorderLanes(object):
    """
        Re-order the "detections" ground truth column vectors in the order of appearance from left to right,
        with respect to the ego vehicle view.
    """
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image, detections = sample['image'], sample['detections']
        
        # Extract the columns that represent valid lane lines
        valid_column_list = [detections[:,i] for i in range(5) if detections[5, i] >= 0.5]
        
        def get_bottom_intercept(col):
            a, b, c, d , y_high = col[0], col[1], col[2], col[3], col[5]
           
            slope = (3 * a * y_high**2) + (2 * b * y_high) + c
            x_high = (a * y_high**3) + (b * y_high**2) + (c * y_high) + d
            x_intercept = x_high + (slope * (1.0 - y_high))
            return x_intercept
        
        
        valid_column_list.sort(key = lambda col: get_bottom_intercept(col))
        
        left_lane_lines = [col for col in valid_column_list if get_bottom_intercept(col) < 0.5]
        right_lane_lines = valid_column_list[len(left_lane_lines):]
        
        x_intercepts = [get_bottom_intercept(col) for col in valid_column_list]

        reordered_detections = np.zeros((6, 2), dtype=np.double)
        
        if len(left_lane_lines) > 0 and len(right_lane_lines) > 0:
            ego_lane_lines = [left_lane_lines[-1], \
                              right_lane_lines[0]]
        else:
            ego_lane_lines = right_lane_lines[:2]

        for i, column in enumerate(ego_lane_lines):
            reordered_detections[:, i] = column.T
            if(column[5] == 0):
                print('intercepts:\n{}\nall lines: {}'.format(x_intercepts, valid_column_list))
                 
        return {'image':image, 'detections':reordered_detections}

    
    
class RandomHorizontalShift(object):
    """
        Shifts the image in the width direction by a small random amount, to crudely simulate lane changes
    """
    def __init__(self, output_width, max_x=10):
        self.max_x = max_x
        self.width = output_width
        
    
    def __call__(self, sample):
        image, detections = sample['image'], sample['detections']
        
        height, width = np.shape(image)[0], np.shape(image)[1]
        
        random_shift = int(self.max_x * np.random.random()) # +/- self.max_x 
        
        shifted_detections = detections
        
        # shift the polynomials
        shifted_detections[3, :] -= random_shift / width
        
        
        # horizontal cropp
        
        shifted_image = image[:, random_shift:(self.width + random_shift), :]
        
        
        
        return {'image': shifted_image, 'detections':shifted_detections}
   