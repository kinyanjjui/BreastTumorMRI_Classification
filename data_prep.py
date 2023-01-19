import pandas as pd
import os
import numpy as np
import pydicom 
from skimage.io import imsave
from tqdm import tqdm

#useful paths
data_path = 'manifest-1662587153455'
boxes_path = 'Annotation_Boxes.csv'
mapping_path = 'Breast-Cancer-MRI-filepath_filename-mapping.csv'
target_png_dir ='png_out'

if not os.path.exists(target_png_dir):
	os.makedirs(target_png_dir)

#loading the bounding boxing  annotation list
print('Loading bounding boxes')
boxes_df = pd.read_csv(boxes_path)
print(f'The shape of the loaded bounded box is{boxes_df.shape()}')
print(boxes_df.head())

#These MRIs are 3-dimensional. Having height(pixels), width(pixels) and
#slice/depth(how far the bounding box/tumour extends in the depths)

#For this project we will only consider the fat-saturated MR exams
#'pre' exams
mapping_df = pd.read_csv(mapping_path)
mapping_df = mapping_df[mapping_df['original_path_and_filename'].str.contains('pre')]

#removing entries from patients that we are not including
#only including patients 201-300
print('Patients 201-300')
crossref_pattern = '|'.join("DICOM_Images/Breast_MRI_{:03d}".format(s) for s in list(range(201, 301)))
# print(crossref_pattern) Should write a test for this 
mapping_df = mapping_df[mapping_df['original_path_and_filename'].str.contains(crossref_pattern)]
print("mapping_df shape:")
print(mapping_df.shape)
#each row in mapping_df represents a different 2d slice of full 3d mri

#Defining the classification task
##all 2d slices containing tumour bounding boxes will be positive(1)
##and all other slices will be negative(0)

#Extracting .png files from the raw DICOM data

def save_dcm_slice(dcm_fname, label, vol_idx):
	"""
	A function that saves the 2d slice .png image of each 3D MRI
	
	Inputs:
	dcm_fname(dir) -- the filename of the source DICOM
	label(int) -- the cancer label of the slice(0,1)
	vol_idx(int)-- patient index in the mri dataset 
	
	Output:None
	"""
	#create a path to save the slice .png file in, according to the original DICOM fname
	png_path = dcm_fname.split('/')[-1].replace('.dcm', f'-{vol_idx}.png') 
	label_dir = 'pos' if label == 1 else 'neg'
	png_path = os.path.join(target_png_dir, label_dir, png_path)

	if not os.path.exists(os.path.join(target_png_dir,label_dir)):
		os.makedirs(os.path.join(target_png_dir,label_dir))

	if not os.path.exists(png_path):
		#only make the png image if it doesn't already exist
		try:
			dcm = pydicom.dcmread(dcm_fname)
		except FileNotFoundError:
		#fix possible errors in filename from list
			dcm_fname_split = dcm_fname.split('/')
			dcm_fname_end = dcm_fname_split[-1]
			assert dcm_fname_end.split('-')[1][0] == '0'

			dcm_fname_end_split = dcm_fname_end.split('-')
			dcm_fname_end ='-'.join([dcm_fname_end_split[0], dcm_fname_end_split[1][1:]])

			dcm_fname_split[-1] = dcm_fname_end
			dcm_fname ='/'.join(dcm_fname_split)
			dcm = pydicom.dcmread(dcm_fname)

	#convert the DICOM into numerical np array of pixel intensity values
	img = dcm.pixel_array

	#convert uint16 datatype to float, scaled properly for uint8
	img = img.astype(np.float) * 255. /img.max()
	#convert from float -> uint8
	img = img.astype(np.uint8)
	#invert image if necessary, according to DICOM metadata
	img_type = dcm.PhotometricInterpretation
	if img_type == 'MONOCHROME1':
		img = np.invert(img)

	#save final .png
	imsave(png_path, img)

print("Balancing and iterating through the patients 3D volumes")
#number of examples for each class
N_class = 2600
#counts of examples extracted from each class
ct_negative = 0
ct_positive = 0

#initialize iteration index of each patient volume
vol_idx = -1
for row_idx, row in tqdm(mapping_df.iterrows(), total=N_class*2):
	#indices start at 1 here
	new_vol_idx = int((row['original_path_and_filename'].split('/')[1]).split('_')[-1])
	slice_idx = int(((row['original_path_and_filename'].split('/')[1]).split('_')[-1]).replace('.dcm',''))

	#new volume: get tumour bounding box
	if new_vol_idx != vol_idx:
		box_row = boxes_df.iloc[[new_vol_idx-1]]
		start_slice = int(box_row['Start Slice'])
		end_slice = int(box_row['End Slice'])
		assert end_slice >= start_slice
	vol_idx = new_vol_idx

	#get DICOM filename
	dcm_fname = str(row['classic_path'])
	dcm_fname = os.path.join(data_path, dcm_fname)


	#determine slice label:
	#(1) if within 3D box, save as positive
	if slice_idx >= start_slice and slice_idx < end_slice:
		if ct_positive >= N_class:
			continue
		save_dcm_slice(dcm_fname, 1, vol_idx)
		ct_positive += 1

	#(2) if outside 3D box by >5 slices, save as negative
	elif (slice_idx + 5) <= start_slice or (slice_idx - 5) > end_slice:
		if ct_negative >= N_class:
			continue
		save_dcm_slice(dcm_fname, 0, vol_idx)
		ct_negative += 1

print('Balancing, done')

#displaying the images
# from random import choice
# positive_image_dir = os.path.join(target_png_dir,'pos')
# negative_img_filenames = os.listdir(positive_image_dir)
# sample_image_path = os.path.join(positive_image_dir, choice(negative_image_filenames))


