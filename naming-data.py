import os

# Function to rename the images in car_data directory which were manually downloaded from Google Images
def rename_files(directory_path, name):
    # List all files in the directory
    files = os.listdir(directory_path)

    #iterator
    i = 1

    # Iterate through each file and rename it
    for file_name in files:
        # Build the full path of the file
        old_path = os.path.join(directory_path, file_name)

        # New file name with  the car name and a number
        new_name = name + str(i) + os.path.splitext(file_name)[1] # Keep the file extension
        
        # Increment the iterator
        i += 1

        # Build the full path of the new file name
        new_path = os.path.join(directory_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {file_name} to {new_name}')



# Specify the directory path where you want to rename files
directory_paths = [
    "./cars_data/convertible",
    "./cars_data/sedan",
    "./cars_data/suv",
    "./cars_data/truck"
]

# Call the function to rename files with last part of the path as the name
for directory_path in directory_paths:
    rename_files(directory_path, directory_path.split('/')[-1])