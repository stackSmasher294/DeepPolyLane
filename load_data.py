import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import numpy as np
import matplotlib.image as mpimg
import cv2
import functools as ft

class TUSimpleLaneDataset(Dataset):
    """ Lane detection dataset """
    
    def __init__(self, json_dataset_list, root_dir, transform=None, degree=3):
        """
        Args:
            json_dataset_list: list of the data points described as JSON strings returned by file.readlines()
        """
        self.json_dataset_list = json_dataset_list
        self.root_dir = root_dir
        self.transform = transform
        self.degree = degree
        
    def __len__(self):
        return len(self.json_dataset_list)
    
    def __getitem__(self, idx):
        image_path, lanes_polyline = self.parse_dataset(self.json_dataset_list[idx], self.root_dir)
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
        
        # columns: 1 for existence probablity, 4 for coefficients of the cubic polynomial f(y) and 2 for y limits
        # number of rows: 5, one for each lane line
        polynomials_and_bounds = np.zeros((7, 5), dtype=np.double) 
        
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
            polynomials_and_bounds[1:, i] = coeffs_and_bounds
            polynomials_and_bounds[0, i] = 1.0 # because it exists
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
        valid_column_list = [detections[:,i] for i in range(5) if detections[0, i] == 1.0]
        
        def get_bottom_intercept(col):
            a, b, c, d , y_high = col[1], col[2], col[3], col[4], col[6]
           
            slope = (3 * a * y_high**2) + (2 * b * y_high) + c
            x_high = (a * y_high**3) + (b * y_high**2) + (c * y_high) + d
            x_intercept = x_high + (slope * (1.0 - y_high))
            return x_intercept
        
        
        valid_column_list.sort(key = lambda col: get_bottom_intercept(col))

        reordered_detections = np.zeros((7, 5), dtype=np.double)

        for i, column in enumerate(valid_column_list):
            reordered_detections[:, i] = column.T
                 
        return {'image':image, 'detections':reordered_detections}

    
    
# class RandomXShift(object):
#     """
#         Shifts the image in the width direction by a small random amount, to crudely simulate lane changes
#     """
   