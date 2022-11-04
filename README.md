# Prompt Generation for Text-to-Image Models Using Deep Reinforcement Learning

## Team Members
Teona Bagashvili <br />
Keith Tyser <br />
Zihan Li

## Code
- FID.py calculates the FID scores between two images.
- scores.py writes the data from scores.txt and unsorted_scores.txt to CSV files.
- stats.py calculates the mean FID score of all ground truth and test image pairs with matching prompts. It also calculates the mean FID score of all ground truth and test image pairs with different prompts. It then does a statistical t-test between the two means of these groups to determine if the difference is statistically significant.

## Data 
- The Ground Truth Images folder contains the ground truth images that were generated using DALL-E mini in our brute force experiment.
- The Test Images folder contains the test images that were generated using DALL-E mini in our brute force experiment.
- scores.txt contains the FID scores calculated between ground truth and test image pairs that had matching prompts.
- unsorted_scores.txt contains the FID scores calculated between ground truth and test image pairs that had different prompts.
- sorted.csv contains the same data from score.txt but in CSV format.
- unsorted.csv contains the same data from unsorted_scores.txt but in CSV format.


## Models
- generate_images.ipynb first generates all of the possible prompts for our brute force experiment. It then feeds all 180 prompts into DALL-E mini and saves the generated images into a google drive directory. We feed the prompts into the model in 36 batches of 5 to avoid out of memory errors due to computational limitations.
