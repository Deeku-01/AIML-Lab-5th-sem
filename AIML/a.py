from queue import PriorityQueue

def print_state(state):
    for row in state:
        print(row)
    print()

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def move_up(state):
    i, j = find_blank(state)
    if i > 0:
        new_state = [row[:] for row in state]  #row[:] shallow copy of the entire row
        new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        return new_state
    return None

def move_down(state):
    i, j = find_blank(state)
    if i < 2:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        return new_state
    return None

def move_left(state):
    i, j = find_blank(state)
    if j > 0:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        return new_state
    return None

def move_right(state):
    i, j = find_blank(state)
    if j < 2:
        new_state = [row[:] for row in state]
        new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        return new_state
    return None

def calculate_heuristics(state, goal_state):
    h = 0  # Initialize heuristic count

    for i in range(3):  # Loop through rows
      for j in range(3):  # Loop through columns
           if state[i][j] != 0 and state[i][j] != goal_state[i][j]:  # Ignore blank space 
                h += 1  # Increment heuristic count
    return h


# A* Algorithm for solving the 8-puzzle
def astar_search(initial_state, goal_state):
    # Priority queue to store (f, g, state) where:
    # f = g + h (total cost)
    # g = number of moves so far
    # h = heuristic value
    pq = PriorityQueue()
    pq.put((0, 0, initial_state))  # Initial state (f=0, g=0, initial_state)
    
    visited_states = set()  # Store visited states to avoid loops
    move_count = 0  # Counter for moves

    while not pq.empty():
        # Get state with lowest f-score
        f, g, current_state = pq.get()
        state_tuple = tuple(map(tuple, current_state))  # Convert to immutable type for storing

        # Print current state
        print_state(current_state)

        # Check if goal is reached
        if current_state == goal_state:
            print(f"Goal reached in {g} moves!")
            return

        # Add current state to visited
        visited_states.add(state_tuple)

        # Generate possible moves
        possible_moves = [
            (move_up(current_state), "UP"),
            (move_down(current_state), "DOWN"),
            (move_left(current_state), "LEFT"),
            (move_right(current_state), "RIGHT")
        ]

        # Process each valid move
        for next_state, move in possible_moves:
            if next_state is not None:  # Check if move is valid
                next_state_tuple = tuple(map(tuple, next_state))  # Convert to tuple for checking
                if next_state_tuple not in visited_states:
                    h = calculate_heuristics(next_state, goal_state)  # Calculate heuristic
                    g_next = g + 1  # Increment move count
                    f_next = g_next + h  # Calculate f-score

                    # Add new state to the priority queue
                    pq.put((f_next, g_next, next_state))

        move_count += 1  # Increment move count

    print("No solution found.")  # If no solution exists


# Initial and goal states
initial_state = [[1, 2, 3], [4, 7, 5], [0, 6, 8]]
goal_state = [[1, 2, 3], [4, 0, 5], [6, 7, 8]]

# Run A* search
astar_search(initial_state, goal_state)
