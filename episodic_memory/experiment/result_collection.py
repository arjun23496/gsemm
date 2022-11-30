import os
import sys

import pickle as pkl

def collect_all_from_files(dirname):
    pkl_files_in_directory = [ f for f in os.listdir(dirname) if f.endswith('.pkl') ]

    # get template of the collated result
    result_collated = {}
    with open(os.path.join(dirname, pkl_files_in_directory[0]), "rb") as f:
        result_collated = { key: [] for key, val in pkl.load(f).items() }

    for pkl_file in pkl_files_in_directory:
        with open(os.path.join(dirname, pkl_file), "rb") as f:
            file_dict = pkl.load(f)

        for key, val in file_dict.items(): # Right now, the collation function can collect only non list values
            if type(val) != list:
                result_collated[key].append(val)
            else:
                print("Lists not supported")
                sys.exit()

    return result_collated

if __name__ == "__main__":
    print(collect_all_from_files("/Users/arjunkaruvally/Downloads/filezilla/final"))
