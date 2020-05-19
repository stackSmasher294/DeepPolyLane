import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

def visualize_detections(image_tensor, detection_tensor, existence_threshold = 0.6):
    batch_size = image_tensor.size()[0]
    print('batch_size: {}'.format(batch_size))
    denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    
    for i in range(batch_size):
        #print('----------------------------------------')
        
        image = denormalize(image_tensor[i]).data.numpy()
        dtectns = detection_tensor[i].data.numpy().reshape((6,2))
        
        # plot the image
        image = np.transpose(image, (1, 2, 0))
        #print('image shape: {}'.format(np.shape(image)))
        fig=plt.figure(figsize=(20, 20), dpi= 100, facecolor='w', edgecolor='k')
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(image)
        
        #plot the detections
        max_num_dtectn = 2
        colors = ['r', 'g', 'b', 'y', 'w']
        for i in range(2):
            line_detection = dtectns[:,i]     
            coeffs, y_bounds = line_detection[0:4], line_detection[4:6]
            image_h, image_w = np.shape(image)[0], np.shape(image)[1]
            #print('image shape: {}, {}'.format(image_h, image_w))
            y_samples = np.linspace(y_bounds[0], y_bounds[1], 50)
            lane_points_x, lane_points_y = [], []
            for y in y_samples:
                a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                x = (a * y**3) + (b * y**2) + (c * y) + d
                lane_points_x.append(x * image_w)
                lane_points_y.append(y * image_h)
            #print('coeff: {}, bounds: {}'.format(coeffs, y_bounds))
#             plt.imshow(image)
            plt.scatter(lane_points_x, lane_points_y, color=colors[i], alpha=0.4, marker='.')
            
    plt.show()

