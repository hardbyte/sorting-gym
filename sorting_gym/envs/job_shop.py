"""Job Shop Scheduling environment with pointer-based neural interface.

Simplified job shop: n jobs × m machines, each job has one operation per machine.
The agent schedules operations to minimize makespan, receiving constant-size
observations via pointer comparisons over a flat list of operations.
"""

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Tuple

from sorting_gym.envs.basic_neural_sort_interface import Instruction
from sorting_gym.envs.combinatorial_base import NeuralCombinatorialInterfaceEnv


class JobShopSchedulingEnv(NeuralCombinatorialInterfaceEnv):
    """Job Shop Scheduling with constant-size observations via pointer comparisons.

    Operations are stored as a flat list. Each operation has a processing time and
    a machine assignment. Pointers index into this flat list.

    Instructions:
        0: ScheduleNext(i) — schedule operation at v[i] at earliest feasible time
        1: MoveVar(i, dir) — move pointer ±1
        2: AssignVar(i, j) — v[i] = v[j]
        3: Finish() — end episode
    """

    def __init__(self, base=20, k=4, starting_min_jobs=2, num_machines=2):
        self.num_machines = num_machines
        self.starting_min_jobs = starting_min_jobs
        self.num_jobs = 0

        # Per operation: processing_time, machine_id
        self.processing_times = None
        self.machine_ids = None
        self.job_ids = None

        # Schedule state
        self.scheduled = None
        self.start_times = None
        self.machine_available_at = None
        self.job_available_at = None
        self.makespan = 0

        instructions = [
            Instruction(0, 'ScheduleNext', Discrete(k),                          self.op_schedule_next),
            Instruction(1, 'MoveVar',      Tuple([Discrete(k), MultiBinary(1)]), self.op_move_var),
            Instruction(2, 'AssignVar',    Tuple([Discrete(k), Discrete(k)]),    self.op_assign_var),
            Instruction(3, 'Finish',       Discrete(1),                          self.op_finish),
        ]

        # 2 comparison attributes: processing_time, machine_id
        super().__init__(
            k=k,
            num_attributes=2,
            instructions=instructions,
            starting_min_items=starting_min_jobs * num_machines,
            base=base,
        )

        self._finished = False
        self.reset()

    @property
    def _num_pointer_features(self):
        # Per pointer: is_scheduled
        return 1

    @property
    def _num_scalar_features(self):
        # Discretized: fraction_scheduled (4 bins)
        return 4

    def _generate_instance(self):
        self.num_jobs = max(2, self.min_num_items // self.num_machines)
        num_ops = self.num_jobs * self.num_machines
        self.num_items = num_ops

        # Generate processing times and machine assignments
        processing_times = []
        machine_ids = []
        job_ids = []
        for j in range(self.num_jobs):
            # Each job visits each machine once, in a random order
            machines = self.np_random.permutation(self.num_machines).tolist()
            for m in machines:
                processing_times.append(int(self.np_random.integers(1, self.base)))
                machine_ids.append(m)
                job_ids.append(j)

        self.processing_times = processing_times
        self.machine_ids = machine_ids
        self.job_ids = job_ids
        self.items = list(zip(processing_times, machine_ids))

        # Schedule state
        self.scheduled = np.zeros(num_ops, dtype=bool)
        self.start_times = np.full(num_ops, -1, dtype=np.int32)
        self.machine_available_at = np.zeros(self.num_machines, dtype=np.int32)
        self.job_available_at = np.zeros(self.num_jobs, dtype=np.int32)
        self.makespan = 0
        self._finished = False

    def _get_pointer_features(self):
        feats = np.zeros((self.k, 1), dtype=np.int8)
        for i in range(self.k):
            feats[i, 0] = self.scheduled[self.v[i]]
        return feats

    def _get_scalar_obs(self):
        obs = np.zeros(4, dtype=np.int8)
        frac = self.scheduled.sum() / max(self.num_items, 1)
        if frac <= 0.25:
            obs[0] = 1
        elif frac <= 0.5:
            obs[1] = 1
        elif frac <= 0.75:
            obs[2] = 1
        else:
            obs[3] = 1
        return obs

    def op_schedule_next(self, args):
        i = args[0] if isinstance(args, (tuple, list)) else args
        op_idx = self.v[i]
        if self.scheduled[op_idx]:
            return  # already scheduled

        job = self.job_ids[op_idx]
        machine = self.machine_ids[op_idx]
        proc_time = self.processing_times[op_idx]

        # Earliest start: max of machine availability and job availability
        earliest = max(self.machine_available_at[machine], self.job_available_at[job])
        self.start_times[op_idx] = earliest
        self.scheduled[op_idx] = True
        end_time = earliest + proc_time
        self.machine_available_at[machine] = end_time
        self.job_available_at[job] = end_time
        self.makespan = max(self.makespan, end_time)

    def op_finish(self, args):
        self._finished = True

    def step(self, action):
        instruction, *args = action
        self.dispatch(instruction, args)

        all_scheduled = self.scheduled.all()
        terminated = self._finished or all_scheduled
        truncated = False
        reward = -1
        if terminated:
            reward += -self.makespan

        self.episode_total_reward += reward
        if terminated:
            self.reward_shortfalls.append(self.episode_total_reward)

        info = {
            'makespan': self.makespan,
            'num_scheduled': int(self.scheduled.sum()),
            'num_operations': self.num_items,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Jobs: {self.num_jobs}, Machines: {self.num_machines}")
        print(f"Operations (proc_time, machine): {self.items}")
        print(f"Scheduled: {list(self.scheduled.astype(int))}")
        print(f"Start times: {list(self.start_times)}")
        print(f"Makespan: {self.makespan}")
        print(f"Pointers: {list(self.v)}")
