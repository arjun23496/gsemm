import os

def add_experiment_id(filename, id):
    """
    Adds a custom experiment id to the filename for unique identification of experiment file
    Parameters
    ----------
    filename
    id : str
        UID of the experiment
    filename : str
        name of the file to add the id

    Returns
    -------
    str
        processed filename
    """

    # this will return a tuple of root and extension
    split_tup = os.path.splitext(filename)
    assert len(split_tup) == 2, "File extension is not correct in '{}'".format(filename)
    return "{}_{}{}".format(split_tup[0], id, split_tup[1])
