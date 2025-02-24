from queue import PriorityQueue

def print_state(state):
    for row in state:
        print(row)
    print()

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j]==0:
                return i,j

def move_up(state):
    i,j=find_blank(state)
    if i>0:
        next_state=[row[:] for row in state]
        next_state[i][j],next_state[i-1][j]=next_state[i-1][j],next_state[i][j]
        return next_state
    return None

def move_down(state):
    i,j=find_blank(state)
    if i<2:
        next_state=[row[:] for row in state]
        next_state[i][j],next_state[i+1][j]=next_state[i+1][j],next_state[i][j]
        return next_state
    return None

def move_left(state):
    i,j=find_blank(state)
    if j>0:
        next_state=[row[:] for row in state]
        next_state[i][j],next_state[i][j-1]=next_state[i][j-1],next_state[i][j]
        return next_state
    return None

def move_right(state):
    i,j=find_blank(state)
    if j<2:
        next_state=[row[:] for row in state]
        next_state[i][j],next_state[i][j+1]=next_state[i][j+1],next_state[i][j]
        return next_state
    return None


def heuristics(cur_state,goal_state):
    h=0
    for i in range(3):
        for j in range(3):
            if cur_state[i][j]!=0 and cur_state[i][j]!=goal_state[i][j]:
                h+=1

    return h


def astar(initital_state,goal_state):
    pq=PriorityQueue()
    pq.put((0,0,initital_state))

    visited_states=set()

    move_count=0

    while not pq.empty():
        f,g,cur_state=pq.get()
        state_tuple=tuple(map(tuple,cur_state))

        print_state(cur_state)

        if(cur_state==goal_state):
            print(f"goal state reached in {g} moves")
            return 
        
        possible_moves=[
            (move_up(cur_state),"UP"),
            (move_down(cur_state),"down"),
            (move_left(cur_state),"left"),
            (move_right(cur_state),"Right")
        ]

        for next_state,move in possible_moves:
            if next_state is not None:
                next_state_tuple=tuple(map(tuple,next_state))
                if next_state_tuple not in visited_states:
                    h= heuristics(next_state,goal_state)
                    g_next=g+1
                    f_next=g_next+h
                    
                    pq.put((f_next,g_next,next_state))

        move_count+=1

    print("NO SOLUTION")


initial_state=[[1,2,3],[4,7,5],[0,6,8]]
goal_state=[[1,2,3],[4,0,5],[6,7,8]]

astar(initial_state,goal_state)
