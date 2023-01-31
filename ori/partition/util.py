import sympy

def split_list(list_t, ratio):
    ratio = float(sympy.Rational(ratio))
    list_1t = list(range(int(len(list_t)*ratio)))
    list_2t = list(range(int(len(list_t)*ratio), len(list_t)))
    list_1 = list(map(list_t.__getitem__, list_1t))
    list_2 = list(map(list_t.__getitem__, list_2t))
    return list_1, list_2
    
def count_dict_elements(dict_t):
    count = 0
    for k in dict_t.keys():
        count = count + len(dict_t[k])
    return count

