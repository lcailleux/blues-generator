def lcp(s, t):
    n = min(len(s), len(t))
    for i in range(0, n):
        if s[i] != t[i]:
            return s[0:i]
    else:
        return s[0:n]


def get_longest_pattern(str):
    lrs = ""
    n = len(str)
    for i in range(0, n):
        for j in range(i + 1, n):
            x = lcp(str[i:n], str[j:n])
            if len(x) > len(lrs):
                lrs = x
    return lrs


def get_latest_tuple_element(tuple_data):
    if isinstance(tuple_data, tuple):
        return tuple_data[-1]
    return ''

