################################
# Helpers to lookup comparisons
###############################


def _get_pairwise_offset(i, j, k):
    i_jmp = 0
    for tmpi in range(i-1, -1, -1):
        i_jmp += k - (tmpi + 1)
    return i_jmp * 6 + (j - i -1) * 6


def v_less_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 0]


def v_greater_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 2]


def v_equals(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 1]


def data_neighbour_greater(obs, i, direction):
    cmps = obs['neighbour_view_comparisons']
    offset = 1 if direction == -1 else 4
    return cmps[8*i + offset]


def data_neighbour_less(obs, i, direction):
    cmps = obs['neighbour_view_comparisons']
    offset = 3 if direction == -1 else 6
    return cmps[8*i + offset]


###########################
# Return instructions
###########################


def SwapWithNext(i):
    return 0, i


def MoveVar(i, direction):
    return 1, i, direction > 0.5


def AssignVar(a, b):
    return 2, a, b

###########################
# Agents
###########################


def bubble_sort_agent(obs, k):
    i, j, l = 0, 1, 2
    if v_less_than(obs, i, j, k):
        if data_neighbour_greater(obs, i, +1):
            return SwapWithNext(i)
        else:
            return MoveVar(i, +1)
    elif v_equals(obs, i, j, k):
        return MoveVar(j, -1)
    else:
        return AssignVar(i, l)


def insertion_sort_agent(obs, k):
    i, j = 0, 1
    low = 2
    # Set initial value of vj
    if v_less_than(obs, i, j, k):
        return AssignVar(j, i)
    elif v_equals(obs, i, j, k):
        return MoveVar(i, +1)
    else:
        if data_neighbour_greater(obs, j, +1):
            return SwapWithNext(j)
        elif v_greater_than(obs, j, low, k) and data_neighbour_less(obs, j, -1):
            return MoveVar(j, -1)
        else:
            return AssignVar(j, i)
