# REFACTORED: Extracted from util.py - converts Gurobi dictionary to list for z variables
def gbdict2lst_z(dic, layer_dims):
    """
    Convert Gurobi z variable dictionary to list format.

    Args:
        dic: Gurobi z variable dictionary
        layer_dims: Layer dimensions (list or int)

    Returns:
        lst: List representation of the dictionary values
    """
    # print(layer_dims)
    # print(dic)
    lst = []
    count = 0
    if isinstance(layer_dims, list) == False:
        for i in range(layer_dims):
            # if dic[i].getAttr('X') is None:
            #     return None
            lst.append(dic[i])
        return lst
    else:
        for dim in layer_dims:
            temp = []
            for i in range(dim):
                temp.append(dic[count][i])
            count += 1
            lst.append(temp)
        return lst