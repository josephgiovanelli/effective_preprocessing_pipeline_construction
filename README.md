# Dependencies

The dependencies can be found in the ```requirements.txt``` file

# Reproducing the experiments

There are two branches related to the DOLAP 2021 paper:
- ```master```, contains the experiments done to discover the dependencies between transformations and, hence, to build effective data pipelines;
- ```preprocessing_impact_experiments```, contains the experiments done to evaluate the impact of data pre-processing.

Run the ```wrapper_experiments.sh``` of these branches to get the results about the related experiments.
The script will create a folder ```results``` where you can find the outcome with some graphs. The folder is not in tracking, hence you can switch between branches, run the script, and have all the results in that folder.

For its extension for the Information Systems journal we created a new branch: ```extension```.
Run the ```wrapper_experiments.sh``` to perform the same experiments of the ```master```, but with the 20 more datasets we added.
Run the ```10x4cv.sh``` to perform the further validation we employed.
