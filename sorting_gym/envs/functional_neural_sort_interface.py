from collections import OrderedDict

import numpy as np
from gym.spaces import Tuple, Discrete, MultiBinary, Dict

from sorting_gym.envs.basic_neural_sort_interface import Instruction
from sorting_gym.envs.sort_interface_base import NeuralSortInterfaceEnv


class FunctionalNeuralSortInterfaceEnv(NeuralSortInterfaceEnv):
    """
    Neural Computer based environment with functions.

    Actions:
    - SwapWithNext(i) which swaps A[v_i] and A[v_i + 1];
    - MoveVar(i, +/- 1) which increments or decrements v_i
      bounded by start and end of the view.
    - AssignVar(i, j) which assigns v_i = v_j
    - FunctionCall(id, l_1, ... l_p, o_1, ..., o_p, r_1, ..., r_q)
    - Return(lp_1, ... lp_q)
    - Swap(i, j) which swaps A[v_i] with A[v_j]
    """

    def __init__(self, base=10, k=4, number_of_functions=2, function_inputs=2, function_returns=1):

        self.number_of_functions = number_of_functions
        self.function_inputs = function_inputs
        self.function_returns = function_returns

        function_space = Tuple([
            # Function ID
            Discrete(number_of_functions), ] +
            # local variables to be assigned
            [Discrete(k)] * function_inputs +
            # outer-scope variables to be passed in
            [Discrete(k)] * function_inputs +
            # return variables
            [Discrete(k)] * function_returns
        )

        instructions = [
            # Instruction(opcode, name, argument space, implementation method)
            Instruction(0, 'SwapWithNext', Discrete(k),                             self.op_swap_with_next),
            Instruction(1, 'MoveVar',      Tuple([Discrete(k), MultiBinary(1)]),    self.op_move_var),
            Instruction(2, 'AssignVar',    Tuple([Discrete(k), Discrete(k)]),       self.op_assign_var),
            Instruction(3, 'FunctionCall', function_space,                          self.op_function_call),
            Instruction(4, 'Return',       Tuple([Discrete(k)] * function_returns), self.op_function_return),
            Instruction(5, 'Swap',         Tuple([Discrete(k), Discrete(k)]),       self.op_swap),
        ]
        # super call will add the action_space attribute
        super().__init__(base, k, instructions)

        self.current_function = -1
        self.previous_action = -1
        self.new_scope = True
        self.previous_args = np.zeros(51, dtype=bool)
        self.call_stack = []

        self.nested_observation_space = Dict(
            pairwise_view_comparisons=MultiBinary((6 * k) * (k-1)//2),
            neighbour_view_comparisons=MultiBinary((4 * k) * 2),
            current_function=Discrete(number_of_functions),
            previous_action=Dict(
                # Which of the 6 instructions was last
                action_type=Discrete(6),
                # Indicate the case of no previous action used at start of episode
                # and on entering a function?
                new_scope=MultiBinary(1),
                arguments=MultiBinary(51)
            ),
        )
        # self.observation_space = flatten_space(self.nested_observation_space)
        self.observation_space = self.nested_observation_space

        self.reset()

    def _get_obs(self):
        k = self.k
        neighbour_comparisons = np.zeros((k, 8), dtype=np.int8)
        pairwise_comparisons = np.zeros((6 * k) * (k - 1) // 2, dtype=np.int8)
        for i in range(k):
            neighbour_comparisons[i, :] = [1, 0, 0, 0, 0, 0, 0, 1]
            if self.v[i] > 0:
                neighbour_comparisons[i, 0] = 0
                neighbour_comparisons[i, 1] = self.A[self.v[i]] > self.A[self.v[i] - 1]
                neighbour_comparisons[i, 2] = self.A[self.v[i]] == self.A[self.v[i] - 1]
                neighbour_comparisons[i, 3] = self.A[self.v[i]] < self.A[self.v[i] - 1]
            if self.v[i] + 1 < len(self.A):
                neighbour_comparisons[i, 4] = self.A[self.v[i]] > self.A[self.v[i] + 1]
                neighbour_comparisons[i, 5] = self.A[self.v[i]] == self.A[self.v[i] + 1]
                neighbour_comparisons[i, 6] = self.A[self.v[i]] < self.A[self.v[i] + 1]
                neighbour_comparisons[i, 7] = 0

            # Pairwise comparisons
            if i < k-1:
                # yeah this is not pretty
                i_jmp = 0
                for tmpi in range(i-1, -1, -1):
                    i_jmp += k - (tmpi + 1)
                i_jmp *= 6

                for j in range(i+1, k):
                    j_stride = j - i - 1
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 0] = self.v[i] < self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 1] = self.v[i] == self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 2] = self.v[i] > self.v[j]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 3] = self.A[self.v[i]] < self.A[self.v[j]]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 4] = self.A[self.v[i]] == self.A[self.v[j]]
                    pairwise_comparisons[i_jmp + (j_stride)*6 + 5] = self.A[self.v[i]] > self.A[self.v[j]]

        # previous action
        # type

        return OrderedDict([
            ('neighbour_view_comparisons', neighbour_comparisons.flatten()),
            ('pairwise_view_comparisons', pairwise_comparisons),
            ('current_function', self.current_function),
            ('previous_action', OrderedDict([
                ('action_type', self.previous_action),
                ('new_scope', [self.previous_action == -1 or self.new_scope]),
                ('arguments', self.previous_args)
            ]))
        ])

    def reset(self):
        super().reset()
        self.current_function = -1
        self.previous_action = -1
        self.call_stack = []
        return self._get_obs()

    def op_function_call(self, args):
        """
        Adds an entry to the call stack to keep track of the variables before the function call,
        the variables that will receive the return values.

        :param args:
            A tuple comprising (id, l_1, ... l_p, o_1, ..., o_p, r_1, ..., r_q)
            Where:
            - l_1, ..., l_p are local variable IDs to be assigned
            - o_1, ..., o_p are the external variable IDs to be passed in
            - r_1, ..., r_q are the external variables IDs to receive the return values

        Call stack is a list of tuples comprising:
            - outer function's id (self.current_function)
            - outer scope's pointers (self.v)
            - return value IDS as a tuple (r_1, ..., r_q)
            - encoding of the function args
        """

        function_id, *function_arguments = args
        assert all(i < self.k for i in function_arguments), "arguments must be in the range 0 to k"
        internal_ids = function_arguments[:self.function_inputs]
        external_ids = self.v[function_arguments[self.function_inputs:2*self.function_inputs]]

        encoded_args = self._encode_function_args(args, offset=4*self.k+1)
        return_values = function_arguments[-self.function_returns:]

        self.call_stack.append([
            self.current_function,
            self.v.copy(),
            return_values,
            encoded_args
        ])
        self.current_function = function_id

        self.v[:] = 0
        self.v[internal_ids] = external_ids

        self.new_scope = True

    def op_function_return(self, return_values):
        outer_function, outer_variables, return_ids, function_encoded_args = self.call_stack.pop()
        # assign the return values to the outer_variables
        outer_variables[return_ids] = self.v[return_values]

        # restore the outer scope
        self.v = outer_variables
        self.current_function = outer_function

        self._function_encoded_args = function_encoded_args

    def op_swap_with_next(self, args):
        # SwapWithNext(i)
        # swaps A[v_i] and A[v_i + 1]
        i = args[0]
        v_i = self.v[i]
        v_i_next = min(len(self.A) - 1, v_i + 1)
        assert v_i < len(self.A), f"Expected v_i ({v_i}) to be less than len(A) ({len(self.A)})"
        assert v_i_next < len(self.A)
        self.A[v_i], self.A[v_i_next] = self.A[v_i_next], self.A[v_i]

    def op_swap(self, args):
        # Swap(i, j)
        # swaps A[v_i] and A[v_j]
        i, j = args
        v_i = self.v[i]
        v_j = self.v[j]

        self.A[v_i], self.A[v_j] = self.A[v_j], self.A[v_i]

    def op_move_var(self, args):
        # MoveVar(i, +/- 1)
        # increments or decrements v_i
        # bounded by start and end of the view.
        i, direction = args
        if direction:
            self.v[i] = min(self.v[i] + 1, len(self.A) - 1)
        else:
            self.v[i] = max(self.v[i] - 1, 0)

    def op_assign_var(self, args):
        # AssignVar(i, j)
        # assigns v_i = v_j
        i, j = args
        self.v[i] = self.v[j]

    def step(self, action):
        instruction, *args = action
        self.new_scope = False
        self.dispatch(instruction, args)

        # Check for solved, calculate reward
        done = self.A == self.tape_env.target
        if done:
            # So the strings get longer
            self.tape_env.episode_total_reward = len(self.A)
        reward = -1
        info_dict = {'data': self.A, 'interface': list(self.v), 'function': self.current_function}

        self.previous_action = instruction

        self._encode_args(instruction, args)
        return self._get_obs(), reward, done, info_dict

    def render(self, mode='human'):
        print(f"""Data: {self.A}
Interface: {list(self.v)}
Current Function: {self.current_function}
Previous Action: {self.previous_action}
""")

    def _encode_args(self, instruction, args):
        """
        Returns an array of 51 booleans corresponding to the arguments
        for the previous instruction.
        """
        self.previous_args *= False

        offset = 0
        if instruction == 0:
            # SwapWithNext Discrete(k)
            # Set the bit at args[0] to True
            self.previous_args[offset + args[0]] = True
        offset += self.k

        if instruction == 1:
            # MoveVar(i, +/- 1) Tuple([Discrete(k), MultiBinary(1)])
            i, direction = args
            dir_offset = self._encode_and_set_discrete_arg(i, offset)
            self.previous_args[dir_offset] = direction
        offset += self.k + 1

        if instruction == 2:
            # AssignVar(i, j) Tuple([Discrete(k), Discrete(k)])
            i, j = args
            tmp_offset = self._encode_and_set_discrete_arg(i, offset)
            self._encode_and_set_discrete_arg(j, tmp_offset)
        offset += 2 * self.k

        if instruction == 3:
            function_encoding = self._encode_function_args(args, offset)
            self.last_function_encoded_args = function_encoding

        function_offset = self._get_function_call_encoding_size()
        offset += function_offset

        if instruction == 4:
            # return Tuple([Discrete(k)] * function_returns)
            ret_offset = offset
            for return_value in args:
                ret_offset = self._encode_and_set_discrete_arg(return_value, ret_offset)

            # pop the last function calls args from the stack.
            f_offset = offset - function_offset
            self.previous_args[f_offset:offset] = self._function_encoded_args

        offset += self.function_returns * self.k

        if instruction == 5:
            # Swap Tuple([Discrete(k), Discrete(k)])
            i, j = args
            tmp_offset = self._encode_and_set_discrete_arg(i, offset)
            self._encode_and_set_discrete_arg(j, tmp_offset)

        offset += self.k * 2

    def _get_function_call_encoding_size(self):
        function_offset = self.number_of_functions + self.k * self.function_inputs * 2 + self.k * self.function_returns
        return function_offset

    def _encode_function_args(self, args, offset=17):
        function_offset = self._get_function_call_encoding_size()
        self.previous_args[offset:offset + function_offset] = False
        # FunctionCall.
        function_id, *function_arguments = args
        # Encode function id
        f_offset = self._encode_and_set_discrete_arg(function_id, offset, self.number_of_functions)
        # Encode Local vars
        internal_ids = function_arguments[:self.function_inputs]
        for value in internal_ids:
            f_offset = self._encode_and_set_discrete_arg(value, f_offset)
        # Encode Nonlocal vars
        external_ids = function_arguments[self.function_inputs:2 * self.function_inputs]
        for value in external_ids:
            f_offset = self._encode_and_set_discrete_arg(value, f_offset)
        # Encode return values
        return_values = function_arguments[-self.function_returns:]
        for value in return_values:
            f_offset = self._encode_and_set_discrete_arg(value, f_offset)

        return self.previous_args[offset:offset + function_offset].copy()

    def _encode_and_set_discrete_arg(self, i, offset, size=None):
        if size is None:
            size = self.k
        self.previous_args[offset + i] = True
        return offset + size
