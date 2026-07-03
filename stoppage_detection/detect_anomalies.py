import cv2
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

def create_mask_hue(video_file_path, is_line_running_path, mask_output_location,
                    target_color, frames_to_use = 200, min_area = 20):

    # Convert the target color to HSV
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)
    target_color_hsv = target_color_hsv[0][0]

    # Define the lower and upper bounds for the target color in HSV color space
    lower_bound = np.array([target_color_hsv[0] - 10,100, 50])
    upper_bound = np.array([target_color_hsv[0] + 10, 255, 255])
    
    # Function to create a mask for each frame where there is color present
    cap = cv2.VideoCapture(video_file_path)

    # use the find mask frames to find frames of normal running to produce the mask
    start_frame = find_mask_frames(is_line_running_path, frames_to_use)
    
    # catch if line is not running at all 
    if start_frame == 'line_not_running':
        return 'Line not running in this video'
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize a binary mask for the total video
    total_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    f = 0
    while cap.isOpened():
        if f < frames_to_use:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create a mask based on the specified color range
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create an empty mask to draw contours
            contour_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            
            # Filter contours by area
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Draw filtered contours on the empty mask
            cv2.drawContours(contour_mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Update the total mask by combining it with the frame mask using bitwise OR
            total_mask = cv2.bitwise_or(total_mask, contour_mask)
        else:
            break
        
        f = f +1
    # Release video capture object
    cap.release()

    #Dilation of mask image
    kernel = np.ones((8,8),np.uint8)
    total_mask_dilation = cv2.dilate(total_mask,kernel,iterations = 1)
    # Create a white image of the same size as the mask
    white_image = np.ones_like(total_mask) * 255

    # Invert the mask by subtracting it from the white image
    inverted_mask = white_image - total_mask_dilation

    # Save the mask as a PNG image
    cv2.imwrite(f'{mask_output_location}/motion_mask.png', inverted_mask)
    return 'Line is running'


def detect_cap_colour(video_path, target_color, mask_path, output_path,
                      line_running_indicator, line_running_time_series, 
                      every_xth_frame = 2, output_video=False, min_area = 40, resize = True):

    if line_running_indicator == 'line_not_running':
        print('line not running in clip provided')
        return
    #Create blank df for outputs
    # Define column names
    columns = ['frame', 'time', 'cap_detected','cap_coordinates']
    # Create blank DataFrame to be populated with data
    cap_detection_df = pd.DataFrame(columns=columns)

    # Convert the target color to HSV
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)
    target_color_hsv = target_color_hsv[0][0]

    # Define the lower and upper bounds for the target color in HSV color space
    lower_bound = np.array([target_color_hsv[0] - 10, 100, 50])
    upper_bound = np.array([target_color_hsv[0] + 10, 255, 255])

    print(line_running_time_series)
    line_running_ts = pd.read_csv(line_running_time_series)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Define the desired width and height for the resized frame
    desired_width = frame_width  # Adjust as needed
    desired_height = frame_height  # Adjust as needed

    if resize:
        # Define the desired width and height for the resized frame
        desired_width = 640  # Adjust as needed
        desired_height = 480  # Adjust as needed

    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_path}/Cap_detected.mp4', fourcc, fps/every_xth_frame, (desired_width, desired_height))

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")

    #Get mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Initialize frame counter
    frame_number = 0
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_number % every_xth_frame == 0:
            if (line_running_ts.loc[line_running_ts['frame'] == frame_number, 'line_running'] == 1).any():

                # Resize the frame
                frame = cv2.resize(frame, (desired_width, desired_height)) 

                # Resize the mask
                mask = cv2.resize(mask, (desired_width, desired_height))

                # Apply the mask to the frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

                # Convert the frame to HSV color space
                hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

                # Threshold the HSV frame to get only the pixels in the specified color range
                mask_colour = cv2.inRange(hsv_frame, lower_bound, upper_bound)

                # Find contours in the mask
                contours, _ = cv2.findContours(mask_colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cap_detected = False
                cap_coords = []

                # Get the current time (in milliseconds)
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Draw bounding boxes around detected regions with area greater than min_area
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        coords = [x,y]
                        cap_coords.append(coords)
                        cap_detected = 1
        
                #Populate data_frame row with frame info
                # Add a row to the DataFrame
                new_row = {'frame': frame_number, 'time': current_time, 'cap_detected': cap_detected, 'cap_coordinates': cap_coords }
                cap_detection_df = pd.concat([cap_detection_df, pd.DataFrame([new_row])], ignore_index=True)
                if output_video:
                    out.write(frame)
        # Increase frame count
        frame_number += 1

    # Release the video capture object and close all windows
    if output_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    cap_detection_df.to_csv(f'{output_path}/motion_detection_output.csv')
    return cap_detection_df

#Detecting caps in a single frame
def detect_cap_colour_frame(frame, target_color, mask_path, min_area = 80):
    
    # Convert the target color to HSV
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)
    target_color_hsv = target_color_hsv[0][0]

    # Define the lower and upper bounds for the target color in HSV color space
    lower_bound = np.array([target_color_hsv[0] - 10, 100, 50])
    upper_bound = np.array([target_color_hsv[0] + 10, 255, 255])

    # Define the desired width and height for the resized frame
    desired_width = 640  # Adjust as needed
    desired_height = 480  # Adjust as neededx``

    #Get mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask
    mask = cv2.resize(mask, (desired_width, desired_height))    
    
    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV frame to get only the pixels in the specified color range
    mask_colour = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_colour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cap_detected = False
    cap_coords = []

    # Increase frame count
    frame_number += 1
    
    # Draw bounding boxes around detected regions with area greater than min_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coords = [x,y]
            cap_coords.append(coords)
            cap_detected = True

    #Populate data_frame row with frame info
    # Add a row to the DataFrame
    new_row = {'frame': frame_number,'cap_detected': cap_detected, 'cap_coordinates': cap_coords }

def motion_mask(video_file, mask_path, min_area = 100):

    #Define min area for colour mask
    min_area = 100
    # Function to create a mask for each frame where there is color present
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize a binary mask for the total video
    total_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Apply background subtraction
        fgmask = fgbg.apply(frame)

        # Apply thresholding to detect significant motion
        _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask to draw contours
        contour_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # Filter contours by area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Draw filtered contours on the empty mask
        cv2.drawContours(contour_mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        # Update the total mask by combining it with the frame mask using bitwise OR
        total_mask = cv2.bitwise_or(total_mask, contour_mask)

    # Release video capture object
    cap.release()

    #Dilation of mask image
    kernel = np.ones((5,5),np.uint8)
    total_mask_dilation = cv2.dilate(total_mask,kernel,iterations = 1)
    # Create a white image of the same size as the mask
    white_image = np.ones_like(total_mask) * 255

    # Invert the mask by subtracting it from the white image
    inverted_mask = white_image - total_mask_dilation

    # Save the mask as a PNG image
    cv2.imwrite(mask_path, inverted_mask)

def motion_detection(video_file_path, mask_file_path, output_path, min_thresh, max_thresh, 
                     line_running_indicator, line_running_time_series, every_xth_frame = 2, show_frame = False, 
                     output_video = False, resize = True):
    import pandas as pd

    if line_running_indicator == 'line_not_runnung':
        print('line not running in clip provided')
        return
    
    # p = 'D:/motion_processed_cam_1/Basler_a2A1920-160ucBAS__40456449__20240305_170324215/timeseries.csv'
    print(line_running_time_series)
    line_running_ts = pd.read_csv(line_running_time_series)    
    # line_running_ts = pd.read_csv(p) 
    # frame_number = 1

    mask_path = mask_file_path
    cap = cv2.VideoCapture(video_file_path)
    
    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    # Define threshold for motion detection
    motion_threshold_min= min_thresh
    motion_threshold_max = max_thresh
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the desired width and height for the resized frame
    desired_width = frame_width  # Adjust as needed
    desired_height = frame_height  # Adjust as needed

    if resize:
        # Define the desired width and height for the resized frame
        desired_width = 640  # Adjust as needed
        desired_height = 480  # Adjust as needed

    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_path}/Motion_detected.mp4', fourcc, fps/every_xth_frame, (desired_width, desired_height))
        out_masked = cv2.VideoWriter(f'{output_path}/Motion_detected_masked.mp4', fourcc, fps/every_xth_frame, (desired_width, desired_height))
    
    #Get mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask
    mask = cv2.resize(mask, (desired_width, desired_height))

    output_dict = {'frame':[],
                'timestamp':[],
                'movement_detected':[],
                'movement_coords':[],
                'location':[],
                'num_contours': []}

    while(cap.isOpened()):
        ret, frame = cap.read()
    
        if not ret:
            break


        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_number % every_xth_frame == 0:
            if (line_running_ts.loc[line_running_ts['frame'] == frame_number, 'line_running'] == 1).any():
                # Resize the frame
                frame = cv2.resize(frame, (desired_width, desired_height)) 

                

                # Convert frame numbers to seconds
                time_in_seconds = timedelta(seconds = frame_number/fps)

                # set motion deetected to 0
                motion_detected = 0
                    
                # Apply the mask to the frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
                # Apply background subtraction
                fgmask = fgbg.apply(masked_frame)
            
                # Apply thresholding to detect significant motion
                _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
            
                # Find contours in the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                contour_centroids_cords = []
                contour_count = 0

                for contour in contours:
                    # Calculate area of each contour
                    area = cv2.contourArea(contour)
                
                    # If the contour area is above the threshold, motion detected
                    if area > motion_threshold_min and area < motion_threshold_max:
                        
                        contour_count += 1

                        # Draw bounding box around the motion area
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Motion Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        centroid_x = (x+(x+w))/2
                        centroid_y = (y+(y+h))/2

                        centroid = [centroid_x, centroid_y]

                        contour_centroids_cords.append(centroid)

                        motion_detected = 1
                
                if output_video:
                    out.write(frame)
                    out_masked.write(masked_frame)
                
                output_dict['frame'].append(frame_number)
                output_dict['timestamp'].append(time_in_seconds)
                output_dict['movement_detected'].append(motion_detected)
                output_dict['movement_coords'].append(contour_centroids_cords)
                output_dict['num_contours'].append(contour_count)

                if show_frame == True:
                    # Draw contours  on the frame
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                    # Resize the frame to make the window smaller
                    # resized_frame = cv2.resize(masked_frame, (640, 480))
                    # Display the resulting frame
                    cv2.imshow('Motion Detection', frame)
                    cv2.imshow('Masked Frame', masked_frame)
                
                    # Press 'q' to exit
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break
                else:
                    continue
        else:
            continue
    
    # Release video capture object and close all windows
    if output_video:
        out.release()
        out_masked.release()
    cap.release()
    cv2.destroyAllWindows()


    import pandas as pd

    df=pd.DataFrame.from_dict(output_dict,orient='index').transpose()

    df.to_csv(f'{output_path}/motion_detection_output.csv')

def is_line_running_motion(video_file_path, min_thresh, max_thresh, output_file_path, 
                           every_xth_frame = 4, line_running_thresh = 40, 
                           show_frame = False):
    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    # Initialize background subtractor
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=5,
    detectShadows=False)


    # Define threshold for motion detection
    motion_threshold_min= min_thresh
    motion_threshold_max = max_thresh

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # initialise the output 
    output_dict = {'frame':[],
                'timestamp':[],
                'contour_count':[],
                'line_running':[]}

    while(cap.isOpened()):
        
        ret, frame = cap.read()

        if not ret:
            break
        
        # Get frame number
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Convert frame numbers to seconds
        time_in_seconds = timedelta(seconds = frame_number/fps)

        # only process evey xth frame to improve performance 
        if frame_number  == 1 or frame_number % every_xth_frame == 0:

            # Resize image 
            frame = cv2.resize(frame, (640, 480))
            # frame = cv2.GaussianBlur(frame, (5, 5), 0)
                    

            # Apply background subtraction
            fgmask = fgbg.apply(frame)

            # Apply thresholding to detect significant motion
            _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
            
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #initalise contour counter
            contour_count = 0
            
            # Loop through all of the contours to check if they are within the min max threshold 
            for contour in contours:
                # Calculate area of each contour
                area = cv2.contourArea(contour)
            
                # If the contour area is above the threshold, motion detected
                if area > motion_threshold_min and area < motion_threshold_max:
                    # If contours fall within range increment counter 
                    contour_count += 1

                    # If we wat to vew the detection draw a rectangle around the detected motion 
                    if show_frame == True:
                        # Draw bounding box around the motion area
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Motion Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            # If the number of contours in range equal or exceed the line running thresh
            # Then we say the line is running 
            if contour_count >= line_running_thresh:
                line_running = 1
            else:
                line_running = 0

            # Add information to the dictionary
            output_dict['frame'].append(frame_number)
            output_dict['timestamp'].append(time_in_seconds)
            output_dict['contour_count'].append(contour_count)
            output_dict['line_running'].append(line_running)
        
        else: 
            # For all other frames add infromation to dictionary 
            output_dict['frame'].append(frame_number)
            output_dict['timestamp'].append(time_in_seconds)
            output_dict['contour_count'].append(contour_count)
            output_dict['line_running'].append(line_running)
            continue

        # Code to display the contours detected on a frame 
        if show_frame == True:
            # Draw contours  on the frame
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            # Resize the frame to make the window smaller
            resized_frame = cv2.resize(frame, (640, 480))
            # Display the resulting frame
            cv2.imshow('Motion Detection', resized_frame)
        
            # Press 'q' to exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Convert dict to df 
    df=pd.DataFrame.from_dict(output_dict,orient='index').transpose()

    # write to csv
    df.to_csv(output_file_path)

def find_mask_frames(data, count=200):

    # Initialize variables to track the longest sequence
    max_count = 0
    current_count = 0
    start_index = -1

    df = pd.read_csv(data)

    if sum(df['line_running']) < count:
        return 'line_not_runnung'

    # Iterate over the DataFrame
    for index, value in enumerate(df['line_running']):
        if value == 1:
            if current_count == 0:
                temp_start_index = index
            current_count += 1
        else:
            current_count = 0
        
        if current_count >= count:
            max_count = current_count
            start_index = temp_start_index
            break  # Stop at the first occurrence of 200 consecutive 1's

    return start_index + 1  # Return +1 because index is zero-based, Excel row is 1-based

def create_mask_motion(video_file_path, is_line_running_path,
                       mask_output_location, frames_to_use = 200, 
                       min_area = 100):
    

    #Define min area for colour mask
    min_area = min_area
    # Function to create a mask for each frame where there is color present
    cap = cv2.VideoCapture(video_file_path)

    # use the find mask frames to find frames of normal running to produce the mask
    start_frame = find_mask_frames(is_line_running_path, frames_to_use)
    
    # catch if line is not running at all 
    if start_frame == 'line_not_runnung':
        return 'Line not running in this video'

    # start_frame = find_mask_frames('movement_test_output.csv', 200)
    # set the video to teh calculated starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize a binary mask for the total video
    total_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # Initialize background subtractor
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=5,
    detectShadows=False)

    
    f = 0
    while cap.isOpened():
        if f < frames_to_use:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            # Apply background subtraction
            fgmask = fgbg.apply(frame)
        
            # Apply thresholding to detect significant motion
            _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            # Create an empty mask to draw contours
            contour_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
            # Filter contours by area
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
            # Draw filtered contours on the empty mask
            cv2.drawContours(contour_mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            if np.mean(contour_mask) > 250:
                continue

            # Update the total mask by combining it with the frame mask using bitwise OR
            total_mask = cv2.bitwise_or(total_mask, contour_mask)
            
            # cv2.imshow('frame',contour_mask)

            # if cv2.waitKey(30) & 0xFF == ord('q'):
            #     break

        else:
            break
        
        f = f +1


    # Release video capture object
    cap.release()
    
    #Dilation of mask image
    kernel = np.ones((5,5),np.uint8)
    total_mask_dilation = cv2.dilate(total_mask,kernel,iterations = 1)
    # Create a white image of the same size as the mask
    white_image = np.ones_like(total_mask) * 255
    
    # Invert the mask by subtracting it from the white image
    inverted_mask = white_image - total_mask_dilation
    
    # Save the mask as a PNG image
    cv2.imwrite(f'{mask_output_location}/motion_mask.png', inverted_mask)
    return 'Line is running'


def detect_motion_stop_in_roi(
    video_file_path,
    roi,
    output_file_path=None,
    every_xth_frame=5,
    min_area=50,
    max_area=1000000,
    tail_seconds=5,
    min_stop_seconds=2,
    stop_motion_ratio=0.1,
    blur_kernel=(5, 5),
    show_frame=False,
):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from video: {video_file_path}")

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError(f"ROI must be (x, y, w, h) with positive width and height. Got: {roi}")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=5,
        detectShadows=False,
    )

    rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number % every_xth_frame != 0:
            continue

        roi_frame = frame[y:y + h, x:x + w]
        if roi_frame.size == 0:
            continue

        if blur_kernel:
            roi_frame = cv2.GaussianBlur(roi_frame, blur_kernel, 0)

        fgmask = fgbg.apply(roi_frame)
        _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_count = 0
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                contour_count += 1
                cx, cy, cw, ch = cv2.boundingRect(contour)
                boxes.append([cx, cy, cw, ch])
                if show_frame:
                    cv2.rectangle(roi_frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)

        motion_detected = 1 if contour_count > 0 else 0
        timestamp_seconds = frame_number / fps

        rows.append({
            "frame": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "motion_detected": motion_detected,
            "num_contours": contour_count,
            "boxes": boxes,
        })

        if show_frame:
            preview = frame.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("ROI Motion Detection", preview)
            cv2.imshow("ROI Threshold", thresh)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows, columns=[
        "frame",
        "timestamp_seconds",
        "motion_detected",
        "num_contours",
        "boxes",
    ])

    if output_file_path is not None:
        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    if df.empty:
        summary = {
            "video_name": Path(video_file_path).name,
            "roi_x": x,
            "roi_y": y,
            "roi_w": w,
            "roi_h": h,
            "frames_scored": 0,
            "tail_frames_scored": 0,
            "tail_motion_ratio": None,
            "tail_avg_contours": None,
            "tail_motion_frames": 0,
            "min_stop_seconds": min_stop_seconds,
            "stop_start_seconds": None,
            "stop_end_seconds": None,
            "stop_duration_seconds": None,
            "stop_motion_ratio": None,
            "stop_detected": None,
        }
        return df, summary

    # Find continuous no-motion runs first. This catches variable-length stops
    # anywhere in the clip, including stops that continue through the final frame.
    stop_start_seconds = None
    stop_end_seconds = None
    stop_duration_seconds = None
    detected_stop_motion_ratio = None
    in_stop_run = False
    run_start_seconds = None
    run_rows = []

    for row in df.itertuples(index=False):
        is_no_motion = row.motion_detected == 0

        if is_no_motion and not in_stop_run:
            in_stop_run = True
            run_start_seconds = float(row.timestamp_seconds)
            run_rows = [row]
        elif is_no_motion:
            run_rows.append(row)
        elif in_stop_run:
            run_end_seconds = float(run_rows[-1].timestamp_seconds)
            run_duration_seconds = run_end_seconds - run_start_seconds
            if run_duration_seconds >= min_stop_seconds:
                stop_start_seconds = run_start_seconds
                stop_end_seconds = run_end_seconds
                stop_duration_seconds = run_duration_seconds
                detected_stop_motion_ratio = 0.0
                break

            in_stop_run = False
            run_start_seconds = None
            run_rows = []

    if stop_start_seconds is None and in_stop_run and run_rows:
        run_end_seconds = float(run_rows[-1].timestamp_seconds)
        run_duration_seconds = run_end_seconds - run_start_seconds
        if run_duration_seconds >= min_stop_seconds:
            stop_start_seconds = run_start_seconds
            stop_end_seconds = run_end_seconds
            stop_duration_seconds = run_duration_seconds
            detected_stop_motion_ratio = 0.0

    # If no pure no-motion run is long enough, allow a little noisy motion
    # within a minimum-duration window.
    if stop_start_seconds is None:
        for window_start_seconds in df["timestamp_seconds"]:
            window_end_seconds = window_start_seconds + min_stop_seconds
            window_df = df[
                (df["timestamp_seconds"] >= window_start_seconds)
                & (df["timestamp_seconds"] <= window_end_seconds)
            ]
            if window_df.empty:
                continue

            window_motion_ratio = float(window_df["motion_detected"].mean())
            if window_motion_ratio <= stop_motion_ratio:
                stop_start_seconds = float(window_start_seconds)
                stop_end_seconds = float(window_df["timestamp_seconds"].max())
                stop_duration_seconds = stop_end_seconds - stop_start_seconds
                detected_stop_motion_ratio = window_motion_ratio
                break

    tail_start_seconds = max(0.0, df["timestamp_seconds"].max() - tail_seconds)
    tail_df = df[df["timestamp_seconds"] >= tail_start_seconds].copy()

    tail_motion_ratio = float(tail_df["motion_detected"].mean()) if not tail_df.empty else None
    tail_avg_contours = float(tail_df["num_contours"].mean()) if not tail_df.empty else None
    tail_motion_frames = int(tail_df["motion_detected"].sum()) if not tail_df.empty else 0
    summary = {
        "video_name": Path(video_file_path).name,
        "roi_x": x,
        "roi_y": y,
        "roi_w": w,
        "roi_h": h,
        "frames_scored": int(len(df)),
        "tail_frames_scored": int(len(tail_df)),
        "tail_motion_ratio": tail_motion_ratio,
        "tail_avg_contours": tail_avg_contours,
        "tail_motion_frames": tail_motion_frames,
        "min_stop_seconds": min_stop_seconds,
        "stop_start_seconds": stop_start_seconds,
        "stop_end_seconds": stop_end_seconds,
        "stop_duration_seconds": stop_duration_seconds,
        "stop_motion_ratio": detected_stop_motion_ratio,
        "stop_detected": int(stop_start_seconds is not None),
    }
    return df, summary
