def modified_main(raw_img_path, ground_truth_img_path, ground_truth_count=None, sort=True):
    """
    Computes deviation of thresholded images using different methods
    compared to user input ground truth image.

    Parameters
    ----------
    raw_img_path : str
        Path of raw image.
    ground_truth_img_path : str
        Path of ground truth image.
    ground_truth_count : int, optional
        User input ground truth cell count.
    sort : bool, optional
        Sort the deviation by majority vote.

    Returns
    -------
    optimal_method : str
        Optimal threshold method.
    optimal_threshold : float
        Threshold given by optimal threshold method.

    """
    pass

