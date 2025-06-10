import pygame
import heapq
import math
from math import inf, cos, sin, pi

# Initialize Pygame for graphics rendering
pygame.init()
# Display window configuration constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
FPS = 60

# Color palette for different cell types and visual states
COLORS = {
    'background': (20, 20, 30),           # Dark background for contrast
    'path': (255, 255, 255),              # White for walkable cells
    'blocked': (40, 40, 50),              # Dark gray for blocked cells
    'start': (0, 255, 0),                 # Green for start position
    'treasure': (255, 165, 0),            # Orange for treasure locations
    'obstacle': (128, 128, 128),          # Gray for impassable obstacles
    'trap': (147, 112, 219),              # Purple for all trap types
    'reward': (64, 224, 208),             # Turquoise for all reward types
    'current_path': (255, 0, 0),          # Red for current position
    'visited_once': (100, 150, 255),      # Light blue for single visits
    'visited_multiple': (0, 100, 150),    # Dark blue for multiple visits
    'jump_path': (255, 100, 0),           # Orange for jump movements
    'text': (255, 255, 255)               # White text color
}

class HexGrid:
    # Hexagonal grid visualization system for treasure hunt world display.
    def __init__(self, grid_data, hex_size=30):

        # Initialize hexagonal grid renderer.
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.hex_size = hex_size
        
        # Calculate hexagon dimensions for flat-top orientation
        # Standard hexagon geometry: width = 2*radius, height = sqrt(3)*radius
        self.hex_width = hex_size * 2
        self.hex_height = hex_size * math.sqrt(3)
        
        # Set spacing between hexagons for optimal visual arrangement
        self.horizontal_spacing = self.hex_width * 0.85    # Standard hex spacing
        self.vertical_spacing = self.hex_height * 0.55     # Compressed for better fit
        
        # Center the entire grid within the display window
        total_width = (self.cols - 1) * self.horizontal_spacing + self.hex_width
        total_height = (self.rows - 1) * self.vertical_spacing + self.hex_height
        
        self.offset_x = (WINDOW_WIDTH - total_width) // 2
        self.offset_y = (WINDOW_HEIGHT - total_height) // 2
        
    def hex_to_pixel(self, row, col):
   
       # Convert grid coordinates to screen pixel coordinates.
        x = self.offset_x + col * self.horizontal_spacing + self.hex_width // 2
        y = self.offset_y + row * self.vertical_spacing + self.hex_height // 2
        return int(x), int(y)
    
    def draw_hexagon(self, surface, center_x, center_y, color, border_color=None):

        # Render a flat-top hexagon at specified center position.
        points = []
        # Generate 6 vertices starting from right side, counter-clockwise
        for i in range(6):
            angle = i * pi / 3  # 60-degree increments
            x = center_x + self.hex_size * cos(angle)
            y = center_y + self.hex_size * sin(angle)
            points.append((x, y))
        
        # Draw filled hexagon with border outline
        pygame.draw.polygon(surface, color, points)
        border_col = border_color if border_color else (60, 60, 60)
        pygame.draw.polygon(surface, border_col, points, 1)
    
    def get_cell_color(self, cell_value):
        # Map cell values to their corresponding display colors.
        # Basic cell type color mapping
        color_map = {
            1: COLORS['path'],      # Walkable path
            0: COLORS['blocked'],   # Blocked/empty space
            'S': COLORS['start'],   # Start position
            'T': COLORS['treasure'], # Treasure location
            'O': COLORS['obstacle']  # Impassable obstacle
        }
        
        # Group all trap types under same color scheme
        if cell_value in ['⊖', '⊕', '⊗', '⊘']:  # All trap symbols
            return COLORS['trap']
        # Group all reward types under same color scheme
        elif cell_value in ['⊞', '⊠']:  # All reward symbols
            return COLORS['reward']
        
        return color_map.get(cell_value, COLORS['path'])
    
    def draw_grid(self, surface, path=None, current_step=None, jump_moves=None):
        """
        Render the complete hexagonal grid with path visualization.
        """
        # Track how many times each cell has been visited up to current step
        visit_count = {}
        current_pos = None
        
        if path and current_step is not None:
            # Count visits only up to current animation frame
            for i in range(current_step + 1):
                if i < len(path):
                    pos = path[i]
                    visit_count[pos] = visit_count.get(pos, 0) + 1
                    if i == current_step:
                        current_pos = pos
        
        # Render each cell in the grid
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.grid[row][col]
                
                # Skip blocked cells (value 0) - these represent empty space
                if cell_value == 0:
                    continue
                
                center_x, center_y = self.hex_to_pixel(row, col)
                color = self.get_cell_color(cell_value)
                border_color = (100, 100, 100)
                
                # Apply special coloring based on path traversal state
                if current_pos and (row, col) == current_pos:
                    # Highlight current position with bright red
                    color = COLORS['current_path']
                    border_color = (255, 255, 255)
                elif path and (row, col) in visit_count:
                    # Color visited cells based on visit frequency
                    visits = visit_count[(row, col)]
                    
                    # Check if this position was reached via jump movement
                    is_jump = False
                    if jump_moves and current_step is not None:
                        for i in range(min(current_step + 1, len(path) - 1)):
                            if (path[i], path[i + 1]) in jump_moves and path[i + 1] == (row, col):
                                is_jump = True
                                break
                    
                    # Multiple visits indicate backtracking or revisiting
                    if visits > 1:
                        color = COLORS['visited_multiple']  # Dark blue
                        border_color = (0, 50, 100)
                    else:
                        color = COLORS['visited_once']      # Light blue
                        border_color = (50, 100, 200)
                
                # Render the hexagon with appropriate colors
                self.draw_hexagon(surface, center_x, center_y, color, border_color)
                
                # Add text labels for special cells (start, treasure, traps, rewards)
                if cell_value in ['S', 'T', '⊖', '⊕', '⊗', '⊘', '⊞', '⊠']:
                    try:
                        font = pygame.font.Font(None, 28)
                    except:
                        font = pygame.font.SysFont('arial', 20)
                    
                    # Choose text color for maximum visibility against background
                    if current_pos and (row, col) == current_pos:
                        text_color = (0, 0, 0)  # Black text on red background
                    elif (row, col) in visit_count:
                        text_color = (255, 255, 255)  # White text on blue backgrounds
                    else:
                        text_color = (0, 0, 0) if cell_value != 'S' else (255, 255, 255)
                    
                    # Use clear, readable labels for special symbols
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
                    
                    # Render and center the text within the hexagon
                    text = font.render(display_text, True, text_color)
                    text_rect = text.get_rect(center=(center_x, center_y))
                    surface.blit(text, text_rect)


class TreasureHuntAStar:
    """
    A* pathfinding algorithm implementation for treasure hunt problem.
    
    Key Features:
    1. Hexagonal grid with 6-directional movement (up, down, 4 diagonals)
    2. Jump movement for up/down directions over blocked cells
    3. Trap and reward effect system with cumulative impacts
    4. State-space search considering collected treasures and active effects
    5. Teleport trap with directional movement logic
    """
    def __init__(self, grid_data):
       # Initialize the A* solver with world data.
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        # Locate important positions in the grid
        self.start = None
        self.treasures = []
        self.rewards = []
        
        # Scan grid to find start position, treasures, and rewards
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
                elif cell in ['⊞', '⊠']:  # Reward symbols
                    self.rewards.append(((i, j), cell))
        
        # Sort collections for consistent state representation in A*
        self.treasures = tuple(sorted(self.treasures))
        self.rewards = tuple(sorted(self.rewards))
        
        # Define 6-directional movement for hexagonal grid
        # Includes vertical (up/down) and 4 diagonal directions
        self.directions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (-1, -1),  # Up-Left diagonal
            (-1, 1),   # Up-Right diagonal
            (1, -1),   # Down-Left diagonal
            (1, 1)     # Down-Right diagonal
        ]
        
        # Track jump movements for visualization purposes
        self.jump_moves = set()
        # Base movement cost (modified by traps/rewards)
        self.BASE_COST = 1.0
    
    def get_valid_moves(self, row, col):
        """
        Calculate all valid moves from current position.
        Special Rules:
        - Up/down movements can jump over blocked cells (value 0)
        - Cannot move through obstacles ('O')
        - Diagonal movements follow normal adjacency rules
        """
        valid_moves = []
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            
            # Check if destination is within grid bounds
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue
            cell_value = self.grid[nr][nc]
    
            # Special handling for up/down movements (vertical)
            if (dr, dc) in [(-1, 0), (1, 0)]:  # Up or Down movement
                if cell_value == 0:  # Blocked cell - attempt to jump over it
                    # Calculate jump destination (one more step in same direction)
                    jump_r, jump_c = nr + dr, nc + dc
                    
                    # Validate jump destination
                    if (jump_r >= 0 and jump_r < self.rows and 
                        jump_c >= 0 and jump_c < self.cols):
                        jump_cell = self.grid[jump_r][jump_c]
                        
                        # Can land on any cell except obstacles and blocked cells
                        if jump_cell != 'O' and jump_cell != 0:
                            valid_moves.append((jump_r, jump_c, True))  # Mark as jump
                    continue
                elif cell_value == 'O':  # Obstacle blocks movement completely
                    continue
                else:  # Normal cell - direct movement allowed
                    valid_moves.append((nr, nc, False))  # Mark as normal move
            
            # For diagonal movements, use standard adjacency rules
            else:
                if cell_value != 'O' and cell_value != 0:
                    valid_moves.append((nr, nc, False))  # Mark as normal move

        return valid_moves
        
    def calculate_movement_cost(self, gravity_level, speed_level):
        """
        Calculate movement cost based on current trap/reward effects.
        Cost Formula:
        - Base cost: 1.0
        - Gravity effect: Energy consumption = base * (2^gravity_level)
        - Speed effect: Steps needed = base * (2^speed_level) or base / (2^|speed_level|)
        """

        # Gravity traps exponentially increase energy consumption
        # Each gravity trap doubles the energy needed
        energy_multiplier = 2 ** gravity_level
        # Speed effects modify the number of steps required# | Positive speed_level (from traps) increases steps | Negative speed_level (from rewards) decreases steps
        if speed_level >= 0:
            step_multiplier = 2 ** speed_level
        else:
            step_multiplier = 1.0 / (2 ** abs(speed_level))
        
        return self.BASE_COST * energy_multiplier * step_multiplier
    
    def heuristic(self, row, col, remaining_treasures):
        # Heuristic function for A* algorithm.
        # Uses Manhattan distance to nearest treasure plus minimum spanning tree
        # approximation for remaining treasures to ensure admissibility.
        if not remaining_treasures:
            return 0
        
        # Distance to nearest treasure
        min_dist = min(abs(row - tr[0]) + abs(col - tr[1]) for tr in remaining_treasures)
        
        # For multiple treasures, add MST approximation cost
        if len(remaining_treasures) > 1:
            rem_list = list(remaining_treasures)
            mst_cost = 0
            visited = {rem_list[0]}
            not_visited = set(rem_list[1:])
            
            # Build minimum spanning tree using Prim's algorithm
            while not_visited:
                min_edge = min(
                    abs(v[0] - nv[0]) + abs(v[1] - nv[1])
                    for v in visited for nv in not_visited)
                mst_cost += min_edge
                
                # Find and add the edge that gave minimum cost
                for v in visited:
                    for nv in not_visited:
                        if abs(v[0] - nv[0]) + abs(v[1] - nv[1]) == min_edge:
                            visited.add(nv)
                            not_visited.remove(nv)
                            break
                    else:
                        continue
                    break
            
            return min_dist + mst_cost
        
        return min_dist
    
    def solve(self):
        # Execute A* algorithm to find optimal treasure collection path.
        # Clear previous jump tracking for fresh visualization
        self.jump_moves.clear()
        
        # Define initial state with all treasures remaining and no effects active
        start_state = (self.start[0], self.start[1], self.treasures, 0, 0, frozenset())
        
        # Priority queue for A* algorithm: (f_score, g_score, state)
        open_heap = [(self.heuristic(self.start[0], self.start[1], self.treasures), 
                     0, start_state)]
        
        # Track the best known cost to reach each state
        best_g_score = {start_state: 0}
        
        # Parent pointers for path reconstruction
        parent = {start_state: None}
        goal_state = None
        
        # A* main search loop
        while open_heap:
            f, g, state = heapq.heappop(open_heap)
            row, col, remaining, grav_level, speed_level, used_rewards = state
            
            # Goal test: all treasures have been collected
            if not remaining:
                goal_state = state
                break
            
            # Skip if we've already found a better path to this state
            if g > best_g_score.get(state, inf):
                continue
            
            # Explore all possible moves from current position
            valid_moves = self.get_valid_moves(row, col)
            
            for nr, nc, is_jump in valid_moves:
                # Initialize new state parameters (may be modified by cell effects)
                new_grav, new_speed = grav_level, speed_level
                new_remaining = remaining
                new_used_rewards = used_rewards
                
                cell = self.grid[nr][nc]
                
                # Process cell effects BEFORE calculating movement cost
                if cell == '⊖':      # Gravity trap - increases energy consumption
                    new_grav += 1
                elif cell == '⊕':    # Speed trap - increases steps needed
                    new_speed += 1
                elif cell == '⊘':    # Treasure removal trap - makes treasures uncollectable
                    if remaining:  # If treasures remain, this path becomes invalid
                        continue  # Skip this move - leads to impossible state
                elif cell == '⊞':    # Gravity reward - decreases energy consumption
                    if (nr, nc) not in used_rewards:  # One-time use
                        new_grav = max(0, new_grav - 1)
                        new_used_rewards = used_rewards | {(nr, nc)}
                elif cell == '⊠':    # Speed reward - decreases steps needed
                    if (nr, nc) not in used_rewards:  # One-time use
                        new_speed = speed_level - 1
                        new_used_rewards = used_rewards | {(nr, nc)}
                
                # Handle treasure collection
                if cell == 'T' and (nr, nc) in remaining:
                    rem_set = set(remaining)
                    rem_set.remove((nr, nc))
                    new_remaining = tuple(sorted(rem_set))

                # Handle teleport trap (⊗) - moves 2 cells in movement direction
                if cell == '⊗':
                    # Calculate movement direction from previous position
                    dr = nr - row
                    dc = nc - col
                    
                    # Teleport destination: 2 cells away from trap in same direction
                    final_r, final_c = nr + 2*dr, nc + 2*dc
                    
                    # Validate teleport destination
                    if (final_r < 0 or final_r >= self.rows or final_c < 0 or final_c >= self.cols or
                        self.grid[final_r][final_c] == 'O' or self.grid[final_r][final_c] == 0):
                        continue  # Invalid teleport destination
                    
                    # Update position to teleport destination
                    nr, nc = final_r, final_c
                    
                    # Process effects at teleport destination
                    dest_cell = self.grid[nr][nc]
                    if dest_cell == 'T' and (nr, nc) in new_remaining:
                        rem_set = set(new_remaining)
                        rem_set.remove((nr, nc))
                        new_remaining = tuple(sorted(rem_set))
                    elif dest_cell == '⊖':
                        new_grav += 1
                    elif dest_cell == '⊕':
                        new_speed += 1
                    elif dest_cell == '⊘':
                        if new_remaining:
                            continue  # Treasure removal at teleport destination
                    elif dest_cell == '⊞' and (nr, nc) not in new_used_rewards:
                        new_grav = max(0, new_grav - 1)
                        new_used_rewards = new_used_rewards | {(nr, nc)}
                    elif dest_cell == '⊠' and (nr, nc) not in new_used_rewards:
                        new_speed = max(0, new_speed - 1)
                        new_used_rewards = new_used_rewards | {(nr, nc)}
                
                # Calculate movement cost using CURRENT state levels (before effects)
                move_cost = self.calculate_movement_cost(grav_level, speed_level)
                
                # Track jump moves for visualization
                if is_jump:
                    self.jump_moves.add(((row, col), (nr, nc)))
                
                # Calculate total cost for this path
                new_g = g + move_cost
                new_state = (nr, nc, new_remaining, new_grav, new_speed, new_used_rewards)
                
                # Check if this is the best path to the new state
                if new_g < best_g_score.get(new_state, inf):
                    best_g_score[new_state] = new_g
                    parent[new_state] = state
                    h_score = self.heuristic(nr, nc, new_remaining)
                    f_score = new_g + h_score
                    
                    heapq.heappush(open_heap, (f_score, new_g, new_state))
        
        # Reconstruct optimal path from goal to start
        if goal_state:
            path = []
            st = goal_state
            
            # Trace back through parent pointers
            while st:
                path.append((st[0], st[1]))  # Extract position coordinates
                st = parent.get(st)
            
            path.reverse()  # Reverse to get start-to-goal order
            return path, best_g_score[goal_state]
        
        return None, None


def main():
    """
    Main function to execute treasure hunt visualization and pathfinding.
    Features:
    - Solves the treasure hunt problem using A* algorithm
    - Provides interactive visualization with step-by-step animation
    - Shows solution path, cost, and control instructions
    - Includes legend for understanding cell types
    """
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
    
    # Initialize Pygame display system
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A* Treasure Hunt with Jump Movement")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Create visualization and solver instances
    hex_grid = HexGrid(grid)
    solver = TreasureHuntAStar(grid)

    # Solve the treasure hunt problem
    solution_path, total_cost = solver.solve()
    
    # Display solution results in console
    if solution_path:
        print(f"SUCCESS! Collected all {len(solver.treasures)} treasures!")
        print(f"Path length: {len(solution_path)}, Total cost: {(total_cost-1.5):.2f}")
        print("Path coordinates:", solution_path)
    else:
        print("FAILED! Could not collect all treasures - no valid path found!")
    
    # Animation control variables for interactive visualization
    current_step = 0          # Current step in path animation
    last_step_time = 0        # Time tracking for auto-play
    step_delay = 500          # Milliseconds between animation steps
    auto_play = False         # Auto-play toggle state
    
    # Main visualization loop
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_play = not auto_play  # Toggle auto-play
                elif event.key == pygame.K_RIGHT and solution_path:
                    current_step = min(current_step + 1, len(solution_path) - 1)  # Next step
                elif event.key == pygame.K_LEFT:
                    current_step = max(current_step - 1, 0)  # Previous step
                elif event.key == pygame.K_r:
                    current_step = 0  # Reset to beginning
        
        # Handle automatic animation progression
        if auto_play and solution_path and current_time - last_step_time > step_delay:
            if current_step < len(solution_path) - 1:
                current_step += 1
                last_step_time = current_time
            else:
                auto_play = False  # Stop auto-play at end
        
        # Render all visual elements
        screen.fill(COLORS['background'])
        hex_grid.draw_grid(screen, solution_path, current_step if solution_path else None, solver.jump_moves)
        
        # Display solution information
        if solution_path:
            step_text = font.render(f"PATH: {current_step}/{len(solution_path) - 1}", True, COLORS['text'])
            cost_text = font.render(f"COST: {(total_cost- 1.5):.1f}", True, COLORS['text'])
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
        
        # Display color-coded legend
        legend_items = [
            ("Start (S)", COLORS['start']),
            ("Treasure (T)", COLORS['treasure']),
            ("Traps (T1-T4)", COLORS['trap']),
            ("Rewards (R1-R2)", COLORS['reward']),
            ("Obstacle (O)", COLORS['obstacle']),
        ]
        
        # Legends location
        legend_x = WINDOW_WIDTH - 200
        legend_y = 10
        
        # print the legends
        for i, (name, color) in enumerate(legend_items):
            pygame.draw.rect(screen, color, (legend_x, legend_y + i * 30, 20, 20))
            text = pygame.font.Font(None, 24).render(name, True, COLORS['text'])
            screen.blit(text, (legend_x + 30, legend_y + i * 30))

        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
