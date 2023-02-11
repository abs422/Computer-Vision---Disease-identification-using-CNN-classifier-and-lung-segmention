# Computer-Vision---Disease-identification-using-CNN-classifier-and-lung-segmention

We primarily ended up implementing two algorithms for predicting segmentation masks of CXR images
– Watershed techniques and U-Net Architecture. The Watershed algorithm was implemented in a step-wise
process which involved binary thresholding, background and foreground extraction, marker extraction and
eventually the segmented mask (See Figure 3). We also identified that there were many traditional and well
regarded methods which didn’t work well for this study. For instance, Otsu thresholding didn’t work well here
and hence we had manually decide the thresholds to do binary thresholding which was one of initial steps in
Watershed segmentation pipeline. In our watershed pipeline, at the end all extracted mask were dilated as it
was noticed that the process was underestimating the mask size. The abrupt boundaries of extracted mask were
also smoothed by the dilation step implemented at the end. The last step of dilation was not required when
masks were predicted using U-Net Architecture. The segmentation algorithm which was developed using Shenzen X-Ray 
set was also tested on Montgomery dataset. The dice coefficients and pixel wise classification metrics were recorded for 
resized masks and ground truth. Once the masks were extracted using both segmentation techniques, the masks were used to
extract the lungs from the original CXR images. These images were then supplied as an input to the CNN
classification model to classify presence/ absence of pulmonary tuberculosis.
