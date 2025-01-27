import importlib
import sys
from time import sleep

code0 = """
def func():
    return 0
"""

code1 = """
def func():
    return 1
"""

code2 = """
def func():
    return 2
"""

# Write the initial code to the file
with open("src/dynamic_import_file.py", "w") as f:
    f.write(code0)
f.close()

sleep(0.1)

# Import the function from the module
import src.dynamic_import_file
importlib.reload(src.dynamic_import_file)  # Reload the module
from src.dynamic_import_file import func
print(func())  # Should print 0

# Modify the code in the file
with open("src/dynamic_import_file.py", "w") as f:
    f.write(code1)

# Reload the module after modifying the code
importlib.reload(src.dynamic_import_file)  # Force reload
from src.dynamic_import_file import func
print(func())  # Should print 1
