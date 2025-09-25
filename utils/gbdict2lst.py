# REFACTORED: Extracted from util.py - converts Gurobi dictionary to list
def gbdict2lst(dic, layer_dims):
    """
    Convert Gurobi variable dictionary to list format.

    Args:
        dic: Gurobi variable dictionary
        layer_dims: Layer dimensions (list or int)

    Returns:
        lst: List representation of the dictionary values
    """
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
            lst.append(dic[i].x)
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i].x)
            count += 1
            lst.append(temp)
        return lst