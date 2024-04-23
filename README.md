# 1. Objectives

Develop an AI model that can classify images as non-referable glaucoma (NRG) or referable glaucoma (RG).

Hypothesis: Machine learning models could detect disease-specific features on color fundus images and use them to predict glaucoma.

# 2. Previous works
## Github Repositories
a. https://github.com/DM2LL/JustRAIGS-IEEE-ISBI-2024

b. https://github.com/kr-viku/GLAUCOMA-DETECTION (AW)

c. https://github.com/ahmed1996said/gardnet?tab=readme-ov-file (<== this is from the "parent" challenge prior to this project that we are working on - contains image preprocessing methods I believe) (AW)

d. https://github.com/Vijaisai/final-year-projrct/blob/master/Glaucoma_Full.ipynb (AW)

e. https://github.com/facebookresearch/dinov2 (ZY)

f.https://github.com/seva100/optic-nerve-cnn

## Papers
a. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8198489/pdf/sensors-21-03704.pdf (ZY)

b. https://ieeexplore.ieee.org/abstract/document/10253652 (AW)

c. https://arxiv.org/abs/2304.07193 (DINOv2) (ZY)

d. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10340208/

e. https://www.sciencedirect.com/science/article/pii/S0039625722001163

f. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9485377/

g. https://ieeexplore.ieee.org/abstract/document/10253652

h. https://scholar.google.com/scholar?as_q=Combination+of+Supervised+Learning+and+Unsupervised+Learning+to+Detect+Ungradable+Images+in+the+Airogs+Challenge&as_occt=title&hl=en&as_sdt=0%2C31

i. https://arxiv.org/abs/1704.00979

# 3. Methodology
## Models

We trained different models through 3 different architectures: ResNet18, ResNet50, and Vision Transformers.

# 4. Experiments
## Data preprocessing
For data preprocessing or image preprocessing, we performed
 ### 1. Directly downsizing all images to a uniform dimension of 512 x 512
 ### 2. Cropping the images to field-of-view, then downsize all to have dimension of 512 x 512
 ### 3. Applying contrast limited adaptive histogram equaliztion of images processed by 2 from above.
## Results
   Project results can be seen in the final presentation slides attached to the end of this README file

# 5. Group Presentation Links
## Initial Results
https://rpiexchange-my.sharepoint.com/:p:/g/personal/fougej_rpi_edu/EXng38SmqXtKi6ja7CABhkEBo3lHTdpViVQqoHcoAZKHEg?e=9Yvxk5

### *Feedback from March 29 presentation*
- Provide more clinical knowledge on glaucoma, for example, what kind of imaging features in retinal fundus images distinguish referable cases from non-referable cases?
- Do a literature review on how related works solve glaucoma diagnosis probelms.
- Refine the research hypothesis to, say, "Machine learning models can detect disease-specific imaging features to accurately classify non-referable and referable glaucoma".
- From Dr. Yan: we are doing blind classification...
### *Additional feedback from Xinrui*
- For ongoing experiments, testing multiple versions of ResNet might not be helpful. If you want to run more experiments, perhaps try ResNet, VGG, AlexNet and see how the performance differs.
- The other team has identified that the cup is the key for classification. My suggestion is that you can manually label the cup bounding box for 30-ish cases and train a network to predict that bounding box, then use the prediction to crop a subregion from the original image. Since the bounding box is but a rough estimate, the task should be very easy, and 30 training samples should be enough.
- You need to do better literature review to find out how others are tackling the problem. 

## Results to share with Dr. Yan
https://rpiexchange-my.sharepoint.com/:p:/g/personal/wanga13_rpi_edu/EafRFpI2YjlKgI9USQSRZTMBjPtHdz6UvJkwSEuXbklpsQ?e=WaVS98

## Final Presention
https://rpiexchange-my.sharepoint.com/:p:/g/personal/fougej_rpi_edu/EQFkajCeoXVBibCj45DdqpEBHn_x_IwkCnB-r4VKNFBAbg?e=8Hhsu7
