import pygame
import sys
import time
import heapq
from math import inf, sqrt, cos, sin, pi

# -----------------------------
# 1. Define the hexagonal grid and parameters
# -----------------------------

# Hexagonal grid dimensions (10 columns x 6 rows)
COLS, ROWS = 10, 6

# Element positions (col, row) - 1-based indexing
POSITIONS = {
    'treasures': [(4, 5), (5, 2), (8, 4), (10, 4)],
    'obstacles': [(1, 4), (3, 3), (4, 4), (5, 3), (5, 5), (7, 4), (7, 5), (8, 5), (9, 2)],
    'rewards': {
        (2, 4): 1, 
        (5, 1): 1,
        (6, 6): 2,
        (8, 3): 2
    },
    'traps': {
        (2, 2): 2,
        (3, 5): 2,
        (4, 2): 4,
        (6, 4): 3,
        (7, 2): 3,
        (9, 3): 1
    },
    'start': (1, 1)
}

# Create grid representation (flipped 180 degrees)
grid = [[ '.' for _ in range(COLS)] for _ in range(ROWS)]

# Convert to 0-based indexing and populate grid (FLIPPED 180 degrees)
# Flip both row and column coordinates
start = (ROWS - POSITIONS['start'][1], COLS - POSITIONS['start'][0])
grid[start[0]][start[1]] = 'S'

for col, row in POSITIONS['treasures']:
    flipped_row = ROWS - row
    flipped_col = COLS - col
    grid[flipped_row][flipped_col] = 'T'

for col, row in POSITIONS['obstacles']:
    flipped_row = ROWS - row
    flipped_col = COLS - col
    grid[flipped_row][flipped_col] = '#'

for (col, row), typ in POSITIONS['rewards'].items():
    flipped_row = ROWS - row
    flipped_col = COLS - col
    grid[flipped_row][flipped_col] = '⊠'

for (col, row), typ in POSITIONS['traps'].items():
    flipped_row = ROWS - row
    flipped_col = COLS - col
    if typ == 1:
        grid[flipped_row][flipped_col] = '⊖'
    elif typ == 2:
        grid[flipped_row][flipped_col] = '⊕'
    elif typ == 3:
        grid[flipped_row][flipped_col] = '⊗'
    elif typ == 4:
        grid[flipped_row][flipped_col] = ''

# Locate all treasures (for pathfinding)
treasures = []
for i in range(ROWS):
    for j in range(COLS):
        if grid[i][j] == 'T':
            treasures.append((i, j))
treasures = tuple(sorted(treasures))

# Debug: Print grid and treasures
print("Grid layout (180° flipped):")
for i, row in enumerate(grid):
    print(f"Row {i}: {' '.join(row)}")
print(f"Start position: {start}")
print(f"Treasures found: {treasures}")

# -----------------------------
# 2. Hexagonal neighbor calculation (FIXED for flat-top orientation)
# -----------------------------
def get_hex_neighbors(row, col):
    """Get valid hexagonal neighbors using odd-r offset coordinates (flat-top hexagons)"""
    neighbors = []
    
    # Hexagonal directions for flat-top hexagons (odd-r offset coordinates)
    if col % 2 == 1:  # Odd column (offset down)
        directions = [
            (-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)
        ]
    else:  # Even column (no offset)
        directions = [
            (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)
        ]
    
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            neighbors.append((nr, nc))
    
    return neighbors

# -----------------------------
# 3. Heuristic (Manhattan + MST) - IMPROVED
# -----------------------------
def heuristic(r, c, remaining):
    """Admissible heuristic: Distance to nearest treasure + MST of remaining treasures."""
    if not remaining:
        return 0
    
    rem_list = list(remaining)
    
    # 1) Distance to nearest treasure (using hex distance approximation)
    def hex_distance(r1, c1, r2, c2):
        # Convert to cube coordinates for proper hex distance
        x1 = c1 - (r1 - (r1 & 1)) // 2
        z1 = r1
        y1 = -x1 - z1
        
        x2 = c2 - (r2 - (r2 & 1)) // 2
        z2 = r2
        y2 = -x2 - z2
        
        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2
    
    dists = [hex_distance(r, c, tr[0], tr[1]) for tr in rem_list]
    h0 = min(dists)
    
    # 2) MST cost among remaining treasures
    mst_cost = 0
    if len(rem_list) > 1:
        visited = {rem_list[0]}
        not_visited = set(rem_list[1:])
        
        # Precompute pairwise distances
        dist_map = {}
        for a in rem_list:
            for b in rem_list:
                if a != b:
                    dist_map[(a, b)] = hex_distance(a[0], a[1], b[0], b[1])
        
        # Prim's algorithm for MST
        while not_visited:
            cand = inf
            next_node = None
            for v in visited:
                for nv in not_visited:
                    d = dist_map[(v, nv)]
                    if d < cand:
                        cand = d
                        next_node = nv
            if next_node is not None:
                mst_cost += cand
                visited.add(next_node)
                not_visited.remove(next_node)
            else:
                break
    
    return h0 + mst_cost

# -----------------------------
# 4. A* Search Implementation (FIXED)
# -----------------------------
def a_star():
    start_state = (start[0], start[1], treasures, 0, 0)
    open_heap = [(heuristic(start[0], start[1], treasures), 0, start_state)]
    best_cost = {start_state: 0}
    parent = {start_state: None}
    
    goal_state = None
    nodes_explored = 0

    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        r, c, rem, grav, spd = state
        nodes_explored += 1
        
        # Goal check
        if not rem:
            goal_state = state
            print(f"Goal found! Nodes explored: {nodes_explored}")
            break
            
        # Skip if a better cost was found
        if g > best_cost.get(state, inf):
            continue
            
        # Expand neighbors using proper hexagonal movement
        for nr, nc in get_hex_neighbors(r, c):
            sym = grid[nr][nc]
            
            # Obstacle check
            if sym == '#':
                continue
                
            # Base move cost
            energy_cost = 2 ** grav
            time_cost = 2 ** spd
            move_cost = energy_cost + time_cost
            new_grav, new_spd = grav, spd
            new_rem = rem
            extra_cost = 0
            
            # Apply cell effects
            if sym == '⊞':  # Trap 1 (gravity)
                new_grav += 1
            elif sym == '⊗':  # Trap 2 (speed)
                new_spd += 1
            elif sym == '⊕':  # Reward 1 (gravity)
                new_grav = max(0, new_grav - 1)
            elif sym == '⊠':  # Reward 2 (speed)
                new_spd = max(0, new_spd - 1)
            elif sym == '⊘':  # Trap 3 (spring) - FIXED
                # Calculate spring direction (same as movement direction)
                dr, dc = nr - r, nc - c
                fr, fc = nr + dr, nc + dc
                
                # Validate forced move
                if 0 <= fr < ROWS and 0 <= fc < COLS and grid[fr][fc] != '#':
                    extra_cost = move_cost
                    nr, nc = fr, fc
                    sym = grid[nr][nc]
                    # Reapply effect at landing
                    if sym == '⊞': 
                        new_grav += 1
                    elif sym == '⊗': 
                        new_spd += 1
                    elif sym == '⊕': 
                        new_grav = max(0, new_grav - 1)
                    elif sym == '⊠': 
                        new_spd = max(0, new_spd - 1)
                else:
                    continue  # Can't make the spring move
            elif sym == '⊖' and rem:  # Trap 4 (block if treasures remain)
                continue
                
            # Treasure collection
            if sym == 'T' and (nr, nc) in rem:
                temp = set(rem)
                temp.remove((nr, nc))
                new_rem = tuple(sorted(temp))
                
            # Calculate new cost and state
            new_g = g + move_cost + extra_cost
            new_state = (nr, nc, new_rem, new_grav, new_spd)
            
            if new_g < best_cost.get(new_state, inf):
                best_cost[new_state] = new_g
                parent[new_state] = state
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc, new_rem), new_g, new_state))

    # Reconstruct path
    path = []
    if goal_state:
        curr = goal_state
        total_cost = best_cost[goal_state]
        while curr:
            path.append((curr[0], curr[1]))
            curr = parent[curr]
        path.reverse()
        print(f"Path found with total cost: {total_cost}")
        print(f"Path length: {len(path)} steps")
    else:
        print("No path found!")
    
    return path

# Compute path with A*
print("Computing path...")
the_path = a_star()
print(f"Computed path: {the_path}")

# -----------------------------
# 5. Hexagonal Visualization with Pygame (IMPROVED)
# -----------------------------
pygame.init()

# Hexagon parameters - flat-top orientation (matching matplotlib)
hex_radius = 30
hex_width = 2 * hex_radius * cos(pi/6)  # Flat-to-flat width  
hex_height = 2 * hex_radius  # Point-to-point height

# Calculate window size with margins
margin = hex_radius
win_width = int(COLS * hex_width + 2 * margin)
win_height = int((ROWS + 0.5) * hex_height + 2 * margin)
screen = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Hexagonal Treasure Hunt (180° Flipped)")
clock = pygame.time.Clock()

# Colors - improved visibility
BACKGROUND = (240, 248, 255)  # Light blue background
color_map = {
    '.': (255, 255, 255),  # Empty (white)
    '#': (64, 64, 64),     # Obstacle (dark grey)
    '⊖': (128, 0, 128),    # Trap 1 (purple)
    '⊕': (128, 0, 128),    # Trap 2 (magenta)
    '⊗': (128, 0, 128),    # Trap 3 (dark magenta)
    '⊘': (128, 0, 128),     # Trap 4 (indigo)
    '⊞': (0, 255, 255),    # Reward 1 (cyan)
    '⊠': (0, 255, 255),    # Reward 2 (deep sky blue)
    'T': (255, 215, 0),    # Treasure (gold)
    'S': (0, 128, 0),      # Start (green)
}

symbol_map = {
    '⊞': '⊞',
    '⊠': '⊠',    # Rewards
    '⊖': '-',
    '⊗': 'x',    # Traps
    '⊕': '+',
    '⊘': '/',    # Spring and Block traps
    '★': '★',
    'S': 'S'     # Treasure and start
}

def hex_to_pixel(row, col):
    """Convert grid coordinates to pixel coordinates (180° flipped)"""
    # Flip the coordinates for display to match the 180° flip
    display_row = ROWS - 1 - row
    display_col = COLS  - col
    
    hex_width = 2 * hex_radius * cos(pi/6)
    hex_height = 2 * hex_radius 
    
    x = margin + display_col * hex_width
    # Odd columns are offset down by half hex height (1-based indexing in matplotlib)
    y_offset = 0 if (display_col + 1) % 2 == 1 else hex_height / 2  # Convert to 1-based for offset check
    y = margin + display_row * hex_height + y_offset
    return (x, y)

def draw_hexagon(surface, center, size, color, symbol=''):
    """Draw a hexagon with flat-top orientation"""
    points = []
    for i in range(6):
        angle_deg = 60 * i  # Flat-top orientation (0° at top-right)
        angle_rad = pi / 180 * angle_deg
        x = center[0] + size * cos(angle_rad)
        y = center[1] + size * sin(angle_rad)
        points.append((x, y))
    
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, (0, 0, 0), points, 2)  # Black border
    
    # Draw symbol if provided
    if symbol:
        font = pygame.font.SysFont('Arial', 16, bold=True)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        text = font.render(symbol, True, text_color)
        text_rect = text.get_rect(center=center)
        surface.blit(text, text_rect)

def draw_grid():
    """Draw the entire hexagonal grid (180° flipped)"""
    screen.fill(BACKGROUND)
    for i in range(ROWS):
        for j in range(COLS):
            center = hex_to_pixel(i, j)
            sym = grid[i][j]
            color = color_map.get(sym, (255, 255, 255))
            symbol = symbol_map.get(sym, '')
            draw_hexagon(screen, center, hex_radius, color, symbol)
            
            # Draw coordinates in original (unflipped) format
            font = pygame.font.SysFont('Arial', 10)
            orig_col = COLS - j
            orig_row = ROWS - i
            coord_text = font.render(f"{orig_col},{orig_row}", True, (100, 100, 100))
            coord_rect = coord_text.get_rect(center=(center[0], center[1] + 15))
            screen.blit(coord_text, coord_rect)

def animate_path():
    """Animate the path through the hexagonal grid"""
    if not the_path:
        print("No path to animate!")
        # Still show the grid
        draw_grid()
        pygame.display.flip()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            time.sleep(0.1)
        return
    
    running = True
    path_index = 0
    
    while running and path_index < len(the_path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
                
        if not running:
            break
            
        # Draw the base grid
        draw_grid()
        
        # Draw the path so far
        current_path = the_path[:path_index+1]
        if len(current_path) > 1:
            path_points = [hex_to_pixel(r, c) for r, c in current_path]
            pygame.draw.lines(screen, (255, 0, 0), False, path_points, 4)  # Red path
        
        # Highlight current position
        current_r, current_c = current_path[-1]
        center = hex_to_pixel(current_r, current_c)
        pygame.draw.circle(screen, (255, 0, 0), (int(center[0]), int(center[1])), hex_radius//2)  # Red dot
        
        # Show step counter
        font = pygame.font.SysFont('Arial', 24, bold=True)
        step_text = font.render(f"Step: {path_index + 1}/{len(the_path)}", True, (0, 0, 0))
        screen.blit(step_text, (10, 10))
        
        pygame.display.flip()
        time.sleep(0.5)  # Slower animation for better visibility
        path_index += 1
    
    # Show completion message
    if the_path and running:
        font = pygame.font.SysFont('Arial', 32, bold=True)
        complete_text = font.render("Path Complete!", True, (0, 128, 0))
        text_rect = complete_text.get_rect(center=(win_width//2, 50))
        screen.blit(complete_text, text_rect)
        pygame.display.flip()
    
    # Wait until window closed
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        time.sleep(0.1)

# Run the animation
animate_path()
pygame.quit()
sys.exit()