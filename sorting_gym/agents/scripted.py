################################
# Helpers to lookup comparisons
###############################


def _get_pairwise_offset(i, j, k):
    i_jmp = 0
    for tmpi in range(i-1, -1, -1):
        i_jmp += k - (tmpi + 1)
    return i_jmp * 6 + (j - i -1) * 6


def data_less_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 3]


def data_greater_than(observation, i, j, k) -> bool:
    cmps = observation['pairwise_view_comparisons']
    offset = _get_pairwise_offset(i, j, k)
    return cmps[offset + 5]


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


def FunctionCall(function_id, local_scope, outer_scope, returns):
    return [3, function_id] + local_scope + outer_scope + returns


def Return(local_scope):
    if not isinstance(local_scope, list):
        local_scope = [local_scope]
    return [4] + local_scope


def Swap(a, b):
    return 5, a, b

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


def debug(*args, **kwargs):
    if True:
        print(*args, **kwargs)


def _last_move_was(previous_action, i, direction, k=4):
    args = previous_action['arguments']
    move_offset = k
    direction = direction > 0.5
    return args[move_offset + i] and (direction == args[move_offset + k])


def _last_swap_was(previous_action, i, j, k=4):
    args = previous_action['arguments']
    offset = k*4+1 + 22 + k
    return args[offset + i] and args[offset + j]


def _last_assign_was(previous_action, i, j, k=4):
    args = previous_action['arguments']
    offset = k*2+1
    return args[offset + i] and args[offset + k + j]


def _is_last_function(previous_action, function_id, k=4):
    args = previous_action['arguments']
    function_offset = k * 4 + 1
    return args[function_offset + function_id]


def _partition_f(obs, k=4):
    i, j, low, high = 0, 1, 2, 3
    previous_action = obs['previous_action']
    if previous_action['new_scope'][0]:
        return AssignVar(i, low)
    elif previous_action['action_type'] == 2 and _last_assign_was(previous_action, i, low):
        return AssignVar(j, low)
    elif v_less_than(obs, j, high, k):
        if previous_action['action_type'] == 5 and _last_swap_was(previous_action, i, j):
            return MoveVar(i, +1)
        elif previous_action['action_type'] == 2 and _last_assign_was(previous_action, j, low) or \
             previous_action['action_type'] == 1 and _last_move_was(previous_action, j, +1) and \
             data_less_than(obs, j, high, k):
            if not v_equals(obs, i, j, k):
                return Swap(i, j)
            else:
                return MoveVar(i, +1)
        else:
            return MoveVar(j, +1)
    elif previous_action['action_type'] == 1 and _last_move_was(previous_action, j, +1):
        return Swap(i, high)
    else:
        return Return(i)


def _quicksort_f(obs, k=4):
    i, j, low, high = 0, 1, 2, 3
    previous_action = obs['previous_action']

    if v_less_than(obs, low, high, k=k):
        # if prev is None then
        if previous_action['new_scope'][0]:
            return FunctionCall(1, [low, high], [low, high], [i])
        # else if prev is call function id 2
        elif previous_action['action_type'] == 3 and _is_last_function(previous_action, 1):
            # todo check args
            return AssignVar(j, i)
        elif previous_action['action_type'] == 2 and _last_assign_was(previous_action, j, i):
            # else if prev = AssignVar
            return MoveVar(i, -1)
        elif previous_action['action_type'] == 1 and _last_move_was(previous_action, i, -1):
            # else if prev = Movevar(i, -1)
            if v_greater_than(obs, i, low, k):
                return FunctionCall(0, [low, high], [low, i], [i])
            else:
                return MoveVar(j, +1)
        elif previous_action['action_type'] == 3 and _is_last_function(previous_action, 0):
            return MoveVar(j, +1)
        elif previous_action['action_type'] == 1 and _last_move_was(previous_action, j, +1) and v_less_than(obs, j, high, k):
            return FunctionCall(0, [low, high], [j, high], [high])
        else:
            return Return(high)
    else:
        return Return(high)


def quicksort_agent(obs, k=4):
    """

    Function 0 will be the entry point immediately calling function 1. `quicksort(low, high)`
    Function 1 will be the recursive quicksort function.
    Function 2 will carry out partitioning. given the index to the pivot, and returns
    the pivot index after partitioning.

    """

    # Set initial value of vj
    function_id = obs['current_function']
    if function_id == -1:
        i, j, low, high = 0, 1, 2, 3
        return FunctionCall(0, [low, high], [low, high], [high])
    elif function_id == 0:
        return _quicksort_f(obs)
    elif function_id == 1:
        return _partition_f(obs)
    else:
        raise ValueError(f"Unexpected function with ID: {function_id}")
