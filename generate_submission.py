import os
import datetime
import shutil

solutions = os.listdir("solutions/")

# Generate solution folder name with timestamp.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
solution_folder = "solutions/" + timestamp + "_solution"

# Copy solution_template folder to new solution folder.
shutil.copytree("solutions/solution_template", solution_folder)

# Copy solution file to new solution folder.
shutil.copy("mysolution.py", solution_folder + "/solution/mysolution.py")

# zip solution folder
shutil.make_archive(solution_folder, 'zip', solution_folder)

# delete solution folder
shutil.rmtree(solution_folder)