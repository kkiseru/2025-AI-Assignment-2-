import pygame
import heapq
import math
import time
from math import inf, cos, sin, pi

# Initialize Pygame
pygame.init()

# Constants for display window
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
FPS = 60

# Color definitions for different cell types and visual states
COLORS = {
    'background': (20, 20, 30),
    'path': (255, 255, 255),
    'blocked': (40, 40, 50),
    'start': (0, 255, 0),
    'treasure': (255, 165, 0),    # Yellow like in reference image
    'obstacle': (128, 128, 128),  # Gray like in reference image
    'trap': (147, 112, 219),      # Purple like in reference image
    'reward': (64, 224, 208),     # Turquoise/Teal like in reference image
    'current_path': (255, 0, 0),
    'visited_once': (100, 150, 255),     # Light blue for single visit
    'visited_multiple': (0, 100, 150),   # Dark blue for multiple visits
    'text': (255, 255, 255)
}

class HexGrid:
    """
    Hexagonal grid visualization class for displaying the treasure hunt world.
    Uses flat-top hexagons arranged in a straight grid pattern for better readability.
    """
    def __init__(self, grid_data, hex_size=30):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.hex_size = hex_size
        
        # Calculate hexagon dimensions for flat-top orientation
        # Width is 2 * size, height is sqrt(3) * size (standard hexagon geometry)
        self.hex_width = hex_size * 2
        self.hex_height = hex_size * math.sqrt(3)
        
        # Spacing between hexagons - reduced for tighter visual arrangement
        self.horizontal_spacing = self.hex_width * 0.85    # Standard hex spacing
        self.vertical_spacing = self.hex_height * 0.55     # Compressed vertical spacing
        
        # Center the grid in the window
        total_width = (self.cols - 1) * self.horizontal_spacing + self.hex_width
        total_height = (self.rows - 1) * self.vertical_spacing + self.hex_height
        
        self.offset_x = (WINDOW_WIDTH - total_width) // 2
        self.offset_y = (WINDOW_HEIGHT - total_height) // 2
        
    def hex_to_pixel(self, row, col):
        """Convert grid coordinates to screen pixel coordinates"""
        x = self.offset_x + col * self.horizontal_spacing + self.hex_width // 2
        y = self.offset_y + row * self.vertical_spacing + self.hex_height // 2
        return int(x), int(y)
    
    def draw_hexagon(self, surface, center_x, center_y, color, border_color=None):
        """Draw a flat-top hexagon at specified center position"""
        points = []
        for i in range(6):
            # Calculate hexagon vertices starting from right side, counter-clockwise
            angle = i * pi / 3
            x = center_x + self.hex_size * cos(angle)
            y = center_y + self.hex_size * sin(angle)
            points.append((x, y))
        
        # Draw filled hexagon and border
        pygame.draw.polygon(surface, color, points)
        border_col = border_color if border_color else (60, 60, 60)
        pygame.draw.polygon(surface, border_col, points, 1)
    
    def get_cell_color(self, cell_value):
        """Map cell values to their corresponding colors"""
        color_map = {
            1: COLORS['path'],
            0: COLORS['blocked'],
            'S': COLORS['start'],
            'T': COLORS['treasure'],
            'O': COLORS['obstacle']
        }
        
        # All trap types use same color
        if cell_value in ['⊖', '⊕', '⊗', '⊘']:
            return COLORS['trap']
        # All reward types use same color
        elif cell_value in ['⊞', '⊠']:
            return COLORS['reward']
        
        return color_map.get(cell_value, COLORS['path'])
    
    def draw_grid(self, surface, path=None, current_step=None):
        """
        Draw the complete hexagonal grid with path visualization.
        Shows visit counts and current position during pathfinding animation.
        """
        # Track visit frequency for each cell up to current step
        visit_count = {}
        current_pos = None
        
        if path and current_step is not None:
            # Count visits only up to current animation step
            for i in range(min(current_step + 1, len(path))):
                pos = path[i]
                visit_count[pos] = visit_count.get(pos, 0) + 1
                if i == current_step:
                    current_pos = pos
        
        # Draw each cell in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.grid[row][col]
                
                # Skip blocked cells (value 0) - these represent empty space
                if cell_value == 0:
                    continue
                
                center_x, center_y = self.hex_to_pixel(row, col)
                color = self.get_cell_color(cell_value)
                border_color = (100, 100, 100)
                
                # Highlight current position in bright red
                if current_pos and (row, col) == current_pos:
                    color = COLORS['current_path']
                    border_color = (255, 255, 255)
                # Color visited cells based on visit frequency
                elif path and (row, col) in visit_count:
                    visits = visit_count[(row, col)]
                    if visits > 1:
                        color = COLORS['visited_multiple']  # Dark blue for multiple visits
                        border_color = (0, 50, 100)
                    else:
                        color = COLORS['visited_once']      # Light blue for single visit
                        border_color = (50, 100, 200)
                
                # Draw the hexagon
                self.draw_hexagon(surface, center_x, center_y, color, border_color)
                
                # Add text labels for special cells
                if cell_value in ['S', 'T', '⊖', '⊕', '⊗', '⊘', '⊞', '⊠']:
                    try:
                        font = pygame.font.Font(None, 28)
                    except:
                        font = pygame.font.SysFont('arial', 20)
                    
                    # Choose text color for visibility
                    if current_pos and (row, col) == current_pos:
                        text_color = (0, 0, 0)  # Black on red background
                    elif (row, col) in visit_count:
                        text_color = (255, 255, 255)  # White on blue backgrounds
                    else:
                        text_color = (0, 0, 0) if cell_value != 'S' else (255, 255, 255)
                    
                    # Use clear labels for special symbols
                    display_text = str(cell_value)
                    symbol_labels = {
                        '⊖': 'T1',  # Gravity trap
                        '⊕': 'T2',  # Speed trap
                        '⊗': 'T3',  # Teleport trap
                        '⊘': 'T4',  # Treasure removal trap
                        '⊞': 'R1',  # Gravity reward
                        '⊠': 'R2'   # Speed reward
                    }
                    display_text = symbol_labels.get(cell_value, str(cell_value))
                    
                    text = font.render(display_text, True, text_color)
                    text_rect = text.get_rect(center=(center_x, center_y))
                    surface.blit(text, text_rect)

class TreasureHuntAStar:
    """
    A* algorithm implementation for treasure hunt pathfinding.
    
    State representation: (row, col, remaining_treasures, gravity_level, speed_level)
    - Position coordinates and remaining treasures to collect
    - Gravity and speed levels affect movement costs
    """
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        # Locate start position and all treasure locations
        self.start = None
        self.treasures = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
        
        # Sort treasures for consistent state representation
        self.treasures = tuple(sorted(self.treasures))
        
        # Movement directions for hexagonal grid (diagonal movements only)
        self.directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
    def heuristic(self, row, col, remaining):
        """
        Admissible heuristic function for A* algorithm.
        
        Combines two components:
        1. Minimum distance from current position to any remaining treasure
        2. Minimum Spanning Tree (MST) cost to connect all remaining treasures
        
        This heuristic is admissible because:
        - It never overestimates the actual cost
        - Manhattan distance is admissible for grid-based movement
        - MST provides lower bound for connecting multiple points
        
        Args:
            row, col: Current position
            remaining: Tuple of remaining treasure coordinates
            
        Returns:
            Estimated cost to collect all remaining treasures
        """
        if not remaining:
            return 0
        
        # Convert to list for easier manipulation
        rem_list = list(remaining)
        
        # Component 1: Minimum Manhattan distance to any remaining treasure
        dists_from_current = [abs(row - tr[0]) + abs(col - tr[1]) for tr in rem_list]
        min_dist = min(dists_from_current)
        
        # Component 2: MST cost for connecting all remaining treasures
        mst_cost = 0
        if len(rem_list) > 1:
            # Prim's algorithm for MST calculation
            visited = {rem_list[0]}
            not_visited = set(rem_list[1:])
            
            # Precompute all pairwise distances
            dist = {}
            for t1 in rem_list:
                for t2 in rem_list:
                    if t1 != t2:
                        dist[(t1, t2)] = abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])
            
            # Build MST by repeatedly adding minimum cost edge
            while not_visited:
                cand_dist, cand_t = inf, None
                for t_in in visited:
                    for t_out in not_visited:
                        if dist[(t_in, t_out)] < cand_dist:
                            cand_dist = dist[(t_in, t_out)]
                            cand_t = t_out
                
                mst_cost += cand_dist
                visited.add(cand_t)
                not_visited.remove(cand_t)
        
        return min_dist + mst_cost
    
    def solve(self):
        """
        Main A* algorithm implementation for treasure hunt problem.
        
        Uses priority queue (heap) to explore states in order of f(n) = g(n) + h(n)
        where g(n) is actual cost and h(n) is heuristic estimate.
        
        Returns:
            Tuple of (optimal_path, total_cost) or (None, None) if no solution
        """
        # Initial state: position + all treasures remaining + initial levels
        start_state = (self.start[0], self.start[1], self.treasures, 0, 0)
        
        # Priority queue: (f_score, g_score, state)
        open_heap = [(self.heuristic(self.start[0], self.start[1], self.treasures), 0, start_state)]
        
        # Track best known cost to reach each state
        best_cost = {start_state: 0}
        
        # Parent tracking for path reconstruction
        parent = {start_state: None}
        
        goal_state = None
        visited_rewards = set()  # Track visited reward cells globally
        
        # A* main loop
        while open_heap:
            f, g, state = heapq.heappop(open_heap)
            row, col, remaining, grav_level, speed_level = state
            
            # Goal test: all treasures collected
            if not remaining and goal_state is None:
                goal_state = state
                break
            
            # Skip if we've found a better path to this state
            if g > best_cost.get(state, inf):
                continue
            
            # Explore all possible moves
            for dr, dc in self.directions:
                nr, nc = row + dr, col + dc
                
                # Boundary and obstacle checks
                if (nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or
                    self.grid[nr][nc] == 'O' or self.grid[nr][nc] == 0):
                    continue
                
                # Calculate movement cost based on current gravity and speed levels
                energy_cost = 2**grav_level  # Gravity affects energy consumption
                time_cost = 2**speed_level   # Speed affects time taken
                move_cost = energy_cost + time_cost
                
                # Initialize new state parameters
                new_grav, new_speed = grav_level, speed_level
                new_remaining = remaining
                extra_cost = 0
                
                cell = self.grid[nr][nc]
                
                # Handle different cell types and their effects
                if cell == '⊖':      # Gravity trap
                    new_grav += 1
                elif cell == '⊕':    # Speed trap
                    new_speed += 1
                elif cell == '⊗':    # Teleport trap - special handling
                    # Teleport moves player 2 additional cells in same direction
                    trap_coord = (nr, nc)
                    
                    # Calculate first teleport destination
                    fr1, fc1 = nr + dr, nc + dc
                    if (fr1 < 0 or fr1 >= self.rows or fc1 < 0 or fc1 >= self.cols or 
                        self.grid[fr1][fc1] == 'O'):
                        continue
                    
                    # Calculate final teleport destination
                    fr2, fc2 = fr1 + dr, fc1 + dc
                    if (fr2 < 0 or fr2 >= self.rows or fc2 < 0 or fc2 >= self.cols or 
                        self.grid[fr2][fc2] == 'O'):
                        continue
                    
                    # Teleport costs 3 moves total (1 + 2 extra)
                    extra_cost = 2 * move_cost
                    
                    # Add intermediate state for path tracking
                    intermediate_state = (trap_coord[0], trap_coord[1], remaining, grav_level, speed_level)
                    if intermediate_state not in best_cost:
                        best_cost[intermediate_state] = g + move_cost
                        parent[intermediate_state] = state
                    
                    # Update position to final teleport destination
                    nr, nc = fr2, fc2
                    cell = self.grid[nr][nc]
                    
                    # Check if teleport destination has treasure
                    if cell == 'T' and (nr, nc) in remaining:
                        rem_set = set(remaining)
                        rem_set.remove((nr, nc))
                        new_remaining = tuple(sorted(rem_set))
                    
                    # Create new state and add to frontier
                    new_state = (nr, nc, new_remaining, new_grav, new_speed)
                    new_g = g + move_cost + extra_cost
                    
                    if new_g < best_cost.get(new_state, inf):
                        best_cost[new_state] = new_g
                        parent[new_state] = intermediate_state
                        heapq.heappush(open_heap, (new_g + self.heuristic(nr, nc, new_remaining), new_g, new_state))
                    continue
                
                elif cell == '⊘':    # Treasure removal trap
                    # Can only move here if all treasures already collected
                    if remaining:
                        continue
                elif cell == '⊞':    # Gravity reward
                    # Decrease gravity level (one-time effect per cell)
                    if (nr, nc) not in visited_rewards:
                        new_grav -= 1
                        visited_rewards.add((nr, nc))
                elif cell == '⊠':    # Speed reward
                    # Decrease speed level (one-time effect per cell)
                    if (nr, nc) not in visited_rewards:
                        new_speed -= 1
                        visited_rewards.add((nr, nc))
                
                # Handle treasure collection
                if cell == 'T' and (nr, nc) in remaining:
                    rem_set = set(remaining)
                    rem_set.remove((nr, nc))
                    new_remaining = tuple(sorted(rem_set))
                
                # Calculate total cost and create new state
                new_g = g + move_cost + extra_cost
                new_state = (nr, nc, new_remaining, new_grav, new_speed)
                
                # Add to frontier if this is the best path to this state
                if new_g < best_cost.get(new_state, inf):
                    best_cost[new_state] = new_g
                    parent[new_state] = state
                    f_score = new_g + self.heuristic(nr, nc, new_remaining)
                    heapq.heappush(open_heap, (f_score, new_g, new_state))
        
        # Reconstruct optimal path from goal to start
        if goal_state:
            path = []
            st = goal_state
            raw_path = []
            
            # Trace back through parent pointers
            while st:
                raw_path.append((st[0], st[1]))
                st = parent.get(st)
            
            raw_path.reverse()
            
            # Remove duplicate consecutive positions (from teleport handling)
            path = [raw_path[0]]
            for i, pos in enumerate(raw_path[1:], 1):
                if pos != raw_path[i-1]:
                    path.append(pos)
            
            return path, best_cost[goal_state]
        
        return None, None

def main():
    """Main function to run the treasure hunt visualization"""
    # Grid layout - 0: blocked, 1: path, S: start, T: treasure, O: obstacle
    # Trap symbols: ⊖(gravity), ⊕(speed), ⊗(teleport), ⊘(treasure removal)
    # Reward symbols: ⊞(gravity), ⊠(speed)
    grid = [
        [   0,   1,   0,   1,   0,   1,   0,   1,   0,   1   ],
        [  'S',  0,   1,   0,  '⊞',  0,   1,   0,   1,   0   ],
        [   0,  '⊕',  0,  '⊘',  0,   1,   0,   1,   0,   1   ],
        [   1,   0,   1,   0,  'T',  0,  '⊗',  0,  'O',  0   ],
        [   0,   1,   0,   1,   0,   1,   0,  '⊠',  0,   1   ],
        [   1,   0,  'O',  0,  'O',  0,   1,   0,  '⊖',  0   ],
        [   0,  '⊞',  0,  'O',  0,  '⊗',  0,  'T',  0,  'T'  ],
        [  'O',  0,   1,   0,   1,   0,  'O',  0,   1,   0   ],
        [   0,   1,   0,  'T',  0,   1,   0,  'O',  0,   1   ],
        [   1,   0,  '⊕',  0,  'O',  0,  'O',  0,   1,   0   ],
        [   0,   1,   0,   1,   0,  '⊠',  0,   1,   0,   1   ],
        [   1,   0,   1,   0,   1,   0,   1,   0,   1,   0   ],
    ]
    
    # Initialize Pygame display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A* Treasure Hunt - Hexagonal Grid (Straight Alignment)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Create visualization and solver objects
    hex_grid = HexGrid(grid)
    solver = TreasureHuntAStar(grid)
    
    # Solve the treasure hunt problem
    print("Solving treasure hunt using A* algorithm...")
    solution_path, total_cost = solver.solve()
    
    if solution_path:
        print(f"Solution found! Path length: {len(solution_path)}, Total cost: {total_cost}")
    else:
        print("No solution found!")
    
    # Animation control variables
    current_step = 0
    last_step_time = 0
    step_delay = 500  # milliseconds between animation steps
    auto_play = False
    
    # Main game loop for visualization
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                elif event.key == pygame.K_RIGHT and solution_path:
                    current_step = min(current_step + 1, len(solution_path) - 1)
                elif event.key == pygame.K_LEFT:
                    current_step = max(current_step - 1, 0)
                elif event.key == pygame.K_r:
                    current_step = 0
        
        # Handle auto-play animation
        if auto_play and solution_path and current_time - last_step_time > step_delay:
            if current_step < len(solution_path) - 1:
                current_step += 1
                last_step_time = current_time
            else:
                auto_play = False
        
        # Render everything
        screen.fill(COLORS['background'])
        hex_grid.draw_grid(screen, solution_path, current_step if solution_path else None)
        
        # Display UI information
        if solution_path:
            step_text = font.render(f"Step: {current_step + 1}/{len(solution_path)}", True, COLORS['text'])
            cost_text = font.render(f"Total Cost: {total_cost:.1f}", True, COLORS['text'])
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
        
        # Show control instructions
        instructions = [
            "SPACE: Auto-play/Pause",
            "LEFT/RIGHT: Step through path",
            "R: Reset to start"
        ]
        
        for i, instruction in enumerate(instructions):
            text = pygame.font.Font(None, 24).render(instruction, True, COLORS['text'])
            screen.blit(text, (10, WINDOW_HEIGHT - 100 + i * 25))
        
        # Display legend
        legend_items = [
            ("Start (S)", COLORS['start']),
            ("Treasure (T)", COLORS['treasure']),
            ("Traps (T1-T4)", COLORS['trap']),
            ("Rewards (R1-R2)", COLORS['reward']),
            ("Obstacle (O)", COLORS['obstacle'])
        ]
        
        legend_x = WINDOW_WIDTH - 200
        legend_y = 10
        
        for i, (name, color) in enumerate(legend_items):
            pygame.draw.rect(screen, color, (legend_x, legend_y + i * 30, 20, 20))
            text = pygame.font.Font(None, 24).render(name, True, COLORS['text'])
            screen.blit(text, (legend_x + 30, legend_y + i * 30))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
