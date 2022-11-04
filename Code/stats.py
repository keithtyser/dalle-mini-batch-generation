import pandas as pd
from scipy.stats import ttest_ind

sorted = pd.read_csv("sorted.csv")
unsorted = pd.read_csv("unsorted.csv")

usorted = unsorted.reset_index()  # make sure indexes pair with number of rows
droped = []
for i, row in unsorted.iterrows():
    if row['original'] == row['test']:
        unsorted= unsorted.drop(i, axis=0)
        droped.append(i)

print(unsorted)
print(droped)

sorted = sorted.drop(droped[0], axis=0)
print(sorted)


sorted_scores = (list(sorted.score))
unsorted_scores = (list(unsorted.score))

stat, p = ttest_ind(sorted_scores, unsorted_scores)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

print(" sorted mean: ", sum(sorted_scores)/len(sorted_scores))
print("unsorted mean: ",  sum(unsorted_scores)/len(unsorted_scores))


