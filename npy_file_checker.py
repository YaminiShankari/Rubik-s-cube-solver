import numpy as np

def inspect_npy_file(file_path):
    try:
        # Pass allow_pickle=True to load the object array
        data = np.load(file_path, allow_pickle=True)

        print(f"File Path: {file_path}")
        print(f"Data Type (dtype): {data.dtype}")
        print(f"Shape: {data.shape}")
        print(f"Number of Dimensions (ndim): {data.ndim}")
        print(f"Total Number of Elements (size): {data.size}")

        # Print a sample of the data, especially for large arrays
        if data.size > 10:
            # Flattening might not work as expected for object arrays
            # Instead, just print the first few elements directly
            print(f"First 5 elements: {data[:5]}")
            print(f"Last 5 elements: {data[-5:]}")
        else:
            print(f"All elements: {data}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading or inspecting the NPY file: {e}")

# Example usage:
inspect_npy_file('captured_faces.npy')