import pickle
import glob

# find a pickle file
pkl_files = glob.glob("../results/*/test*/buffer/*.pkl")
if pkl_files:
    print(f"Found {len(pkl_files)} pickle files")
    print(f"Checking: {pkl_files[0]}")
    
    with open(pkl_files[0], 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nType: {type(data)}")
    
    if isinstance(data, list):
        print(f"Length: {len(data)}")
        if data:
            print(f"First element type: {type(data[0])}")
            if hasattr(data[0], '__len__') and len(data[0]) > 0:
                print(f"First element length: {len(data[0])}")
                print(f"First element[0] type: {type(data[0][0])}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    else:
        print(f"Data: {data}")