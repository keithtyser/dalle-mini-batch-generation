# Prompt Generation for Text-to-Image Models Using Deep Reinforcement Learning

## Team Members
Teona Bagashvili <br />
Keith Tyser

## Code 
- FID.py calculates the FID score between ground truth images and test images.
- stats.py calculates the mean FID score between matching and unmatching ground truth images and test images. It also calculates a statistical T-test between the means of these two groups.

## Data
- scores.txt is the FID scores calculated between matching pairs of ground truth and test images.
- unsorted_scores.txt is the FID scores calculated between pairs that don't match of ground truth and test images.
- sorted.csv contains the same data as scores.txt but in CSV format.
- unsorted.csv contains the same data as unsorted_scores.txt but in CSV format.

## Models
- generate_images.ipynb first creates all of the possible prompts for the brute force experiment we ran. It then feeds all of these prompts into DALL-E mini and saves them to a google drive directory. We split the 180 prompts into 36 batches of 5 before feeding them into DALL-E mini to avoid out of memory errors.
