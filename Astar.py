import heapq
from math import inf

# Define the grid as a list of strings or 2D array. 
# with various traps, rewards, obstacles, and treasures:
grid = [
	[   0,   1,   0,   1,   0,   1,   0,   1,   0,   1   ],	# 1 = PATH , 0 = Blocked
	[  'S',  0,   1,   0,  '⊞',  0,   1,   0,   1,   0   ],	# ⊖ = Trap1
	[   0,  '⊕',  0,  '⊘',  0,   1,   0,   1,   0,   1   ],	# ⊕ = Trap2
	[   1,   0,   1,   0,  'T',  0,  '⊗',  0,  'O',  0   ],	# ⊗ = Trap3
	[   0,   1,   0,   1,   0,   1,   0,  '⊠',  0,   1   ],	# ⊘ = Trap4
	[   1,   0,  'O',  0,  'O',  0,   1,   0,  '⊖',  0   ],	# T = Treasure
	[   0,  '⊞',  0,  'O',  0,  '⊗',  0,  'T',  0,  'T'  ],	# ⊞ = Reward1
	[  'O',  0,   1,   0,   1,   0,  'O',  0,   1,   0   ],	# ⊠ = Reward2
	[   0,   1,   0,  'T',  0,   1,   0,  'O',  0,   1   ],	# S = Starting Position
	[   1,   0,  '⊕',  0,  'O',  0,  'O',  0,   1,   0   ],	# O = Obstacle
	[   0,   1,   0,   1,   0,  '⊠',  0,   1,   0,   1   ],	# 0 = blocked
	[   1,   0,   1,   0,   1,   0,   1,   0,   1,   0   ],
]

# Find grid dimensions
rows, cols = len(grid), len(grid[0])

# Identify start and treasure locations
start = None
treasures = []
for i in range(rows):
    for j in range(cols):
        cell = grid[i][j]
        if cell == 'S':
            start = (i, j)
        elif cell == 'T':
            treasures.append((i, j))
# Convert treasures list to a tuple (to use as part of state key)
treasures = tuple(sorted(treasures))

# State representation: (row, col, remaining_treasures, gravity_level, speed_level).
# gravity_level and speed_level are integers representing the exponent for gravity/speed modifiers:
#    effective energy cost per step = 2^gravity_level, 
#    effective step (time) cost per move = 2^speed_level.
# For example, gravity_level=1 means energy cost is doubled (Trap1 active), 
# speed_level=-1 means step cost is halved (Reward2 active), etc.
from math import fabs

def heuristic(row, col, remaining):
    """Admissible heuristic: Manhattan distance from current cell to nearest treasure 
       + MST (minimum spanning tree) length over all remaining treasures."""
    # If no remaining treasures, heuristic is 0 (goal already reached).
    if not remaining:
        return 0
    # Convert remaining treasures to list of coordinates for convenience
    rem_list = list(remaining)
    # Compute Manhattan distance from current position to each remaining treasure
    dists_from_current = [abs(row - tr[0]) + abs(col - tr[1]) for tr in rem_list]
    min_dist = min(dists_from_current)  # nearest treasure distance
    # Compute MST among remaining treasures using Prim's algorithm
    mst_cost = 0
    if len(rem_list) > 1:
        # Start MST from the first treasure
        visited = {rem_list[0]}
        not_visited = set(rem_list[1:])
        # Precompute Manhattan distances between every pair of treasures for efficiency
        dist = {}
        for t1 in rem_list:
            for t2 in rem_list:
                if t1 != t2:
                    dist[(t1, t2)] = abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])
        # Build MST connecting all remaining treasures
        while not_visited:
            # find the shortest edge from visited to not_visited
            cand_dist, cand_t = inf, None
            for t_in in visited:
                for t_out in not_visited:
                    if dist[(t_in, t_out)] < cand_dist:
                        cand_dist = dist[(t_in, t_out)]
                        cand_t = t_out
            # include that edge
            mst_cost += cand_dist
            visited.add(cand_t)
            not_visited.remove(cand_t)
    # Total heuristic is nearest-treasure distance + MST cost
    return min_dist + mst_cost

# Initialize data structures for A* search
start_state = (start[0], start[1], treasures, 0, 0)  # start with no traps active (levels 0)
open_heap = [(heuristic(start[0], start[1], treasures), 0, start_state)]  # (f, g, state)
# Dictionaries to track best discovered cost and parent state for path reconstruction
best_cost = {start_state: 0}
parent = {start_state: None}

# Allowed movements (N, S, E, W)
directions = [ (-1, -1), (-1, 1), (1, -1), (1, 1) ]

# A* Search loop
goal_state = None
visited_rewards = set()
while open_heap:
    f, g, state = heapq.heappop(open_heap)
    row, col, remaining, grav_level, speed_level = state
    # If this state has all treasures collected, we found a solution
    if not remaining and goal_state is None:
        goal_state = state
        break

    # Skip if we have already found a better way to this state
    if g > best_cost.get(state, inf):
        continue
    # Expand neighbors
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        # Check bounds and obstacles
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if grid[nr][nc] == 'O' or grid[nr][nc] == 0:  # obstacle cell
            continue
        # Determine cost of moving into (nr, nc)
        # Current movement cost components
        energy_cost = 2**grav_level   # energy per step
        time_cost   = 2**speed_level  # steps (time) per move
        move_cost = energy_cost + time_cost
        new_grav, new_speed = grav_level, speed_level
        new_remaining = remaining
        extra_cost = 0  # extra cost if Trap3 triggers an additional move
        # Check cell type and apply effects
        cell = grid[nr][nc]
        if cell == '⊖':      # Trap 1
            new_grav += 1  # gravity doubled

        elif cell == '⊕':    # Trap 2
            new_speed += 1  # speed halved (steps doubled)

        elif cell == '⊗':    # Trap 3
            trap_coord = (nr, nc)
            # First forced move
            fr1, fc1 = nr + dr, nc + dc
            if fr1 < 0 or fr1 >= rows or fc1 < 0 or fc1 >= cols or grid[fr1][fc1] == 'O':
                 continue

            # Second forced move
            fr2, fc2 = fr1 + dr, fc1 + dc
            if fr2 < 0 or fr2 >= rows or fc2 < 0 or fc2 >= cols or grid[fr2][fc2] == 'O':
                continue
            
            # Apply cost for the extra moves
            extra_cost = 2 * move_cost
            intermediate_state = (trap_coord[0], trap_coord[1], remaining, grav_level, speed_level)
            if intermediate_state not in best_cost:
                best_cost[intermediate_state] = g + move_cost
                parent[intermediate_state] = state

            nr, nc = fr2, fc2
            cell = grid[nr][nc]

            new_remaining = remaining
            if cell == 'T' and (nr, nc) in remaining:
                rem_set = set(remaining)
                rem_set.remove((nr, nc))
                new_remaining = tuple(sorted(rem_set))

            new_state = (nr, nc, new_remaining, new_grav, new_speed)
            new_g = g + move_cost + extra_cost

            if new_g < best_cost.get(new_state, inf):
                best_cost[new_state] = new_g
                parent[new_state] = intermediate_state
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc, new_remaining), new_g, new_state))
                continue


        elif cell == '⊘':    # Trap 4
            # If treasures remain, stepping here ends the hunt (invalid path)
            if remaining:
                continue
                #new_remaining = ()          # will end it when it steps int trap 3 as it clears all the uncollected treassure
            # If no treasures remaining, it's effectively harmless at end.
            
        elif cell == '⊞':    # Reward 1
            if (nr, nc) not in visited_rewards:
                new_grav -= 1  # gravity halved
                visited_rewards.add((nr, nc))

        elif cell == '⊠':    # Reward 2
            if(nr, nc) not in visited_rewards:
                new_speed -= 1  # speed doubled (half time per move)
                visited_rewards.add((nr, nc))
        # Check for treasure collection on the landing cell (if any)

        if cell == 'T':
            if (nr, nc) in remaining:
                # Remove this treasure from the remaining set
                rem_set = set(remaining)
                rem_set.remove((nr, nc))
                new_remaining = tuple(sorted(rem_set))
        # Compute new state's cost g'
        new_g = g + move_cost + extra_cost
        new_state = (nr, nc, new_remaining, new_grav, new_speed)

        # If we found a cheaper way to this state, record it and push to open list
        if new_g < best_cost.get(new_state, inf):
            best_cost[new_state] = new_g
            parent[new_state] = state
            # Push with priority f = g + h
            heapq.heappush(open_heap, (new_g + heuristic(nr, nc, new_remaining), new_g, new_state))

# Reconstruct path if goal reached
path = []
if goal_state:
    # 1. rebuild the path exactly as before
    st, raw_path = goal_state, []
    while st:
        raw_path.append((st[0], st[1]))
        st = parent.get(st)
    raw_path.reverse()

    # 2. trim immediate duplicates (optional; keeps table tidy)
    path = [raw_path[0]] + [p for i, p in enumerate(raw_path[1:], 1)
                             if p != raw_path[i-1]]

    # 3. summary block (feel free to keep / drop)
    print("\n======= TREASURE HUNT SUMMARY =======")
    print(f" Treasures collected : {len(treasures)}")
    print(f" Physical moves      : {len(path) - 1}")
    print(f" Total cost          : {best_cost[goal_state]:.1f}\n")

    # 4. table header (only three columns now)
    print(f"{'Step':>4} │ {'Position (r,c)':^15} │ {'Cell Type'}")
    print("─────┼─────────────────┼────────────")

    # helper to name each symbol
    def classify(val):
        if val == 'S': return 'Start'
        if val == 'T': return 'Treasure'
        if val in ('⊞','⊠'):      return 'Reward'
        if val in ('⊖','⊕','⊗','⊘'): return 'Trap'
        return 'Normal'

    # 5. rows
    for i, (r, c) in enumerate(path):
        kind = classify(grid[r][c])
        print(f"{i:>4} │ {' ' * 4}({r:>2},{c:>2}){' ' * 5}│ {kind}")

else:
    print("No solution found.")

