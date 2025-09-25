# REFACTORED: Extracted from util.py - gets index from cumulative probability list
def get_index_from_prob_list(prob_list, num):
    """
    Get index from cumulative probability list based on random number.

    Args:
        prob_list: Cumulative probability list
        num: Random number for selection

    Returns:
        i: Index of selected item, or None if not found
    """
    for i in range(len(prob_list)):
        if num <= prob_list[i]:
            return i