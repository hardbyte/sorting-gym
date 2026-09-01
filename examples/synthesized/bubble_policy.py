"""A sorting policy written by a language model, kept verbatim.

Produced by `examples/llm_synthesis.py` with qwen3.6:9b's larger sibling
qwen3.6:27b, as the first candidate of the first round -- one generation, no
refinement. Scored during the run on lengths 5, 10 and 20 only.

Measured afterwards on lengths it was never scored on, 10 instances each:

    n      solved   mean steps   steps/n^2   bubble_sort_agent
    5      10/10          13.2       0.528                12.9
    10     10/10          76.3       0.763                62.7
    20     10/10         340.9       0.852               277.7
    50     10/10        2450.4       0.980              1786.1
    100    10/10       10326.2       1.033              7126.0
    200    10/10       43291.4       1.082             28631.8

It is bubble sort, anchoring v2 at the left edge and using AssignVar(0, 2) to
restart each pass -- the same use of the third pointer as the paper's
`bubble_sort_agent`. It runs about 1.5x longer because it rescans the whole
array every pass instead of shrinking the unsorted boundary.

That it generalizes to 200 is a consequence of being correct rather than a
separate finding: the observation never reveals the array's length, so there is
nothing about length to overfit. The result worth noting is that a frozen model
recovered a correct pointer-machine sorting algorithm from a comparison-only
interface in a single sample.

Kept exactly as generated, including the comments where the model reaches for
an accessor that cannot work and then corrects itself.
"""

def policy(facts):
    # We have three pointers v0, v1, v2.
    # At start: v0 and v2 at index 0, v1 at last index (len-1).
    # Goal: sort A in place using swaps of adjacent elements and pointer moves.
    # Strategy: Use a bubble-sort-like approach where we move pointers to find inversions and swap them.
    
    # Key idea: 
    # - v0 can scan from left to right, checking if current element > next element (inversion).
    # - v1 can scan from right to left, similarly.
    # - We need a coordinated strategy. 
    
    # Let's think of a simple strategy:
    # 1. If v0 is not at the right edge and A[v0] > A[v0+1], swap them. This pushes larger elements right.
    # 2. After swapping or if no inversion, move v0 right by one.
    # 3. Once v0 reaches the end, we can reset or use other pointers.
    
    # But we have three pointers. Let's use a classic bubble sort pass with v0:
    # - Move v0 from left to right, swapping adjacent inversions.
    # - After one full pass (v0 at right edge), the largest element is at the end.
    # - Then we can reset v0 to start and repeat, but we need to detect when sorted.
    
    # However, we don't know when it's sorted directly. The episode ends automatically when sorted.
    # We just need to make progress toward sorting.
    
    # Simple approach: 
    # - If v0 is not at the right edge and A[v0] > A[v0+1], swap.
    # - Else, move v0 to the right.
    # - When v0 reaches the right edge, we need to reset. We can use AssignVar to reset v0 to 0's position? 
    #   But we don't have a fixed pointer at 0. However, initially v2 is also at 0. We could keep v2 at 0 as an anchor?
    
    # Let's try: Keep v2 anchored at the left edge (index 0) by checking facts.at_left_edge(2). If not, move it left until it is.
    # Use v0 to scan from v2's position to the right, swapping inversions.
    # When v0 reaches the right edge, reset v0 to v2's position (AssignVar(0, 2)).
    
    # Step 1: Ensure v2 is at left edge. If not, move it left.
    if not facts.at_left_edge(2):
        return "MoveVar(2, -1)"
    
    # Step 2: Check if v0 is at the right edge. If so, reset v0 to v2's position (start of array).
    if facts.at_right_edge(0):
        return "AssignVar(0, 2)"
    
    # Step 3: Check if there's an inversion at v0: A[v0] > A[v0+1]
    if facts.data_greater_than(0, 0) and not facts.at_right_edge(0):
        # Wait, data_greater_than(i, j) compares A[v_i] and A[v_j]. 
        # To compare A[v0] and A[v0+1], we need to know v0+1's value. But we don't have a direct way.
        # However, facts.data_neighbour_greater(0, +1) tells us if A[v0] > A[v0+1].
        pass
    
    # Use data_neighbour_greater for checking inversion at v0
    if not facts.at_right_edge(0):
        if facts.data_neighbour_greater(0, 1):
            return "SwapWithNext(0)"
    
    # If no inversion to swap, move v0 right
    return "MoveVar(0, +1)"
