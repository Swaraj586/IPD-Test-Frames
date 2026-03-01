## 1. Extracting Image Frames
- Use the image_extract.py script to capture individual frames from a source video.
- Execution: Run ```bash 
python image_extract.py.```
- Configuration: Before running, update the script to point to your specific Input video path and your desired Output frames directory.

## 2. Dataset Labeling
Once the frames are gathered, organize them into folders (or apply labels) based on the specific navigation decision they represent:

- left
- right
- center
- stop

## 3. Accuracy Prediction
- Use the acc_test.py script to evaluate the system's performance against your labeled dataset.
- Execution: Run ```bash python acc_test.py.```
- Configuration: Ensure you update the dataset folder location within the script to point to your labeled frame data.
- Result: The script will calculate and output the final accuracy percentage of the system.
