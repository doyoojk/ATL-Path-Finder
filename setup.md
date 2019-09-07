# Setup
Clone this repository:
`git clone https://github.gatech.edu/omscs6601/assignment_2.git`

If you run across certificate authentication issues during the clone, set the git SSL Verify option to false: `git config --global http.sslVerify false`.

## Python Dependencies

The submission scripts depend on the presence of 5 python packages - `requests`, `future`, `nelson`, `networkx`, and `matplotlib`. If you are missing any of these packages, install them from the online Python registries. The easiest way to do this is through pip:

`pip install -r requirements.txt`

## Keeping your code up to date
After the clone, we recommend creating a branch and developing your agents on that branch:

`git checkout -b develop`

(assuming develop is the name of your branch)

Should the TAs need to push out an update to the assignment, commit (or stash if you are more comfortable with git) the changes that are unsaved in your repository:

`git commit -am "<some funny message>"`

Then update the master branch from remote:

`git pull origin master`

This updates your local copy of the master branch. Now try to merge the master branch into your development branch:

`git merge master`

(assuming that you are on your development branch)

There are likely to be merge conflicts during this step. If so, first check what files are in conflict:

`git status`

The files in conflict are the ones that are "Not staged for commit". Open these files using your favourite editor and look for lines containing `<<<<` and `>>>>`. Resolve conflicts as seems best (ask a TA if you are confused!) and then save the file. Once you have resolved all conflicts, stage the files that were in conflict:

`git add -A .`

Finally, commit the new updates to your branch and continue developing:

`git commit -am "<funny message vilifying TAs for the update>"`

# Submit your code
A friendly reminder: please ensure that your submission is in `submission.py`. The script described in the following section automatically sends that file to the servers for processing.

To submit your code and have it evaluated for a grade, use `python submit.py`. We are going to limit you to 2 submissions per half hour (subject to change depending on load on servers), and the last submission before the deadline will be used to determine your grade.
