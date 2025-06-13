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
        self.hex_width = hex_size * 2
        self.hex_height = hex_size * math.sqrt(3)
        
        # Set spacing between hexagons
        self.horizontal_spacing = self.hex_width * 0.85
        self.vertical_spacing = self.hex_height * 0.55
        
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
        for i in range(6):
            angle = i * pi / 3
            x = center_x + self.hex_size * cos(angle)
            y = center_y + self.hex_size * sin(angle)
            points.append((x, y))
        
        pygame.draw.polygon(surface, color, points)
        border_col = border_color if border_color else (60, 60, 60)
        pygame.draw.polygon(surface, border_col, points, 1)
    
    def get_cell_color(self, cell_value):
        # Map cell values to their corresponding display colors.
        color_map = {
            1: COLORS['path'],
            0: COLORS['blocked'],
            'S': COLORS['start'],
            'T': COLORS['treasure'],
            'O': COLORS['obstacle']
        }
        
        if cell_value in ['⊖', '⊕', '⊗', '⊘']:
            return COLORS['trap']
        elif cell_value in ['⊞', '⊠']:
            return COLORS['reward']
        
        return color_map.get(cell_value, COLORS['path'])
    
    def draw_grid(self, surface, path=None, current_step=None, jump_moves=None):
        # Render the complete hexagonal grid with path visualization.
        visit_count = {}
        current_pos = None
        
        if path and current_step is not None:
            for i in range(current_step + 1):
                if i < len(path):
                    pos = path[i]
                    visit_count[pos] = visit_count.get(pos, 0) + 1
                    if i == current_step:
                        current_pos = pos
        
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.grid[row][col]
                
                if cell_value == 0:
                    continue
                
                center_x, center_y = self.hex_to_pixel(row, col)
                color = self.get_cell_color(cell_value)
                border_color = (100, 100, 100)
                
                if current_pos and (row, col) == current_pos:
                    color = COLORS['current_path']
                    border_color = (255, 255, 255)
                elif path and (row, col) in visit_count:
                    visits = visit_count[(row, col)]
                    is_jump = False
                    if jump_moves and current_step is not None:
                        for i in range(min(current_step + 1, len(path) - 1)):
                            if (path[i], path[i + 1]) in jump_moves and path[i + 1] == (row, col):
                                is_jump = True
                                break
                    
                    if visits > 1:
                        color = COLORS['visited_multiple']
                        border_color = (0, 50, 100)
                    else:
                        color = COLORS['visited_once']
                        border_color = (50, 100, 200)
                
                self.draw_hexagon(surface, center_x, center_y, color, border_color)
                
                if cell_value in ['S', 'T', '⊖', '⊕', '⊗', '⊘', '⊞', '⊠']:
                    try:
                        font = pygame.font.Font(None, 28)
                    except:
                        font = pygame.font.SysFont('arial', 20)
                    
                    if current_pos and (row, col) == current_pos:
                        text_color = (0, 0, 0)
                    elif (row, col) in visit_count:
                        text_color = (255, 255, 255)
                    else:
                        text_color = (0, 0, 0) if cell_value != 'S' else (255, 255, 255)
                    
                    display_text = str(cell_value)
                    symbol_labels = {
                        '⊖': 'T1',
                        '⊕': 'T2',
                        '⊗': 'T3',
                        '⊘': 'T4',
                        '⊞': 'R1',
                        '⊠': 'R2'
                    }
                    display_text = symbol_labels.get(cell_value, str(cell_value))
                    
                    text = font.render(display_text, True, text_color)
                    text_rect = text.get_rect(center=(center_x, center_y))
                    surface.blit(text, text_rect)


class TreasureHuntAStar:
    """
    A* pathfinding algorithm implementation for treasure hunt problem.
    Modified to prioritize least number of steps rather than least cost.
    """
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        self.start = None
        self.treasures = []
        self.rewards = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
                elif cell in ['⊞', '⊠']:
                    self.rewards.append(((i, j), cell))
        
        self.treasures = tuple(sorted(self.treasures))
        self.rewards = tuple(sorted(self.rewards))
        
        self.directions = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
        
        self.jump_moves = set()
        self.BASE_COST = 1.0
    
    def get_valid_moves(self, row, col):
        valid_moves = []
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                continue
            cell_value = self.grid[nr][nc]
    
            if (dr, dc) in [(-1, 0), (1, 0)]:
                if cell_value == 0:
                    jump_r, jump_c = nr + dr, nc + dc
                    if (jump_r >= 0 and jump_r < self.rows and
                        jump_c >= 0 and jump_c < self.cols):
                        jump_cell = self.grid[jump_r][jump_c]
                        if jump_cell != 'O' and jump_cell != 0:
                            valid_moves.append((jump_r, jump_c, True))
                    continue
                elif cell_value == 'O':
                    continue
                else:
                    valid_moves.append((nr, nc, False))
            
            else:
                if cell_value != 'O' and cell_value != 0:
                    valid_moves.append((nr, nc, False))

        return valid_moves
    
    def calculate_movement_cost(self, gravity_level, speed_level):
        # We'll keep the cost calculation but prioritize steps in our heuristic
        if speed_level >= 0:
            step_multiplier = 2 ** speed_level
        else:
            step_multiplier = 1.0 / (2 ** abs(speed_level))
        
        return self.BASE_COST * (2 ** gravity_level) * step_multiplier
    
    def estimate_trap_cost(self, trap_type, current_grav, current_speed):
        """Estimate the cost impact of stepping on a trap"""
        trap_effects = {
            '⊖': (1, 0),    # Gravity increase
            '⊕': (0, 1),    # Speed increase
            '⊗': (0, 0),    # Teleport - hard to estimate
            '⊘': (0, 0)     # Block if treasures remain
        }
        
        grav_change, speed_change = trap_effects[trap_type]
        new_grav = current_grav + grav_change
        new_speed = current_speed + speed_change
        
        # Calculate how this affects future move costs
        return self.calculate_movement_cost(new_grav, new_speed) * 3  # Estimate 3 moves affected

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def heuristic(self, row, col, remaining_treasures):
        """Heuristic that prioritizes steps (distance to nearest treasure)"""
        if not remaining_treasures:
            return 0
            
        min_dist = inf
        for treasure in remaining_treasures:
            dist = self.manhattan_distance((row, col), treasure)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def solve(self):
        self.jump_moves.clear()
        
        # Start with initial gravity and speed
        start_state = (self.start[0], self.start[1], self.treasures, 0, 0, 0)  # Added step count
        
        # Priority queue now prioritizes (steps + heuristic, steps, state)
        open_heap = [(self.heuristic(self.start[0], self.start[1], self.treasures),
                    0,  # steps
                    start_state)]
        
        best_g_score = {start_state: 0}
        best_steps = {start_state: 0}  # Track best steps to reach each state
        parent = {start_state: None}
        state_history = {start_state: (0, 0)}  # Now only tracks (gravity, speed)
        goal_state = None
        
        while open_heap:
            _, steps, state = heapq.heappop(open_heap)
            row, col, remaining, grav_level, speed_level, _ = state
            
            if not remaining:
                goal_state = state
                break
            
            if steps > best_steps.get(state, inf):
                continue
            
            valid_moves = self.get_valid_moves(row, col)
            
            for nr, nc, is_jump in valid_moves:
                new_grav, new_speed = grav_level, speed_level
                new_remaining = remaining
                new_steps = steps + 1
                
                cell = self.grid[nr][nc]
                
                # Handle traps
                if cell == '⊖':
                    new_grav += 1
                elif cell == '⊕':
                    new_speed += 1
                elif cell == '⊘':
                    if remaining:
                        continue
                elif cell == '⊗':
                    # Modified teleport trap behavior - move backwards 3 steps
                    path_history = []
                    current_state = state
                    for _ in range(3):
                        if current_state is None:
                            break
                        path_history.append((current_state[0], current_state[1]))
                        current_state = parent.get(current_state)
                    
                    if len(path_history) >= 3:
                        nr, nc = path_history[2]
                        
                        if (nr >= 0 and nr < self.rows and nc >= 0 and nc < self.cols and
                            self.grid[nr][nc] != 'O' and self.grid[nr][nc] != 0):
                            
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
                                    continue
                            elif dest_cell == '⊞':
                                new_grav -= 1  # Apply reward effect directly
                            elif dest_cell == '⊠':
                                new_speed = max(0, new_speed - 1)  # Apply reward effect directly
                        else:
                            continue
                    else:
                        continue
                
                # Apply rewards but don't let them create loops
                elif cell == '⊞':
                    new_grav -= 1  # Apply every time
                elif cell == '⊠':
                    new_speed = max(0, new_speed - 1)  # Apply every time
                
                # Treasure collection
                if cell == 'T' and (nr, nc) in remaining:
                    rem_set = set(remaining)
                    rem_set.remove((nr, nc))
                    new_remaining = tuple(sorted(rem_set))
                
                new_state = (nr, nc, new_remaining, new_grav, new_speed, new_steps)
                
                # Only consider this state if it improves our step count
                if new_steps < best_steps.get(new_state, inf):
                    best_steps[new_state] = new_steps
                    best_g_score[new_state] = best_g_score[state] + self.calculate_movement_cost(grav_level, speed_level)
                    parent[new_state] = state
                    state_history[new_state] = (new_grav, new_speed)
                    h_score = self.heuristic(nr, nc, new_remaining)
                    
                    # Priority is now (steps + heuristic, steps, state)
                    heapq.heappush(open_heap, (new_steps + h_score, new_steps, new_state))
        
        if goal_state:
            path = []
            states = []
            st = goal_state
            while st:
                path.append((st[0], st[1]))
                states.append(st)
                st = parent.get(st)
            path.reverse()
            states.reverse()
            
            # Extract gravity and speed at each step
            step_states = []
            for i, state in enumerate(states):
                if i == 0:
                    step_states.append((0, 0))  # Initial state
                else:
                    step_states.append(state_history[state])
            
            return path, best_g_score[goal_state], step_states
        
        return None, None, None

def main():
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
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A* Treasure Hunt - Least Steps Priority")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    hex_grid = HexGrid(grid)
    solver = TreasureHuntAStar(grid)

    solution_path, total_cost, step_states = solver.solve()
    
    if solution_path:
        print(f"SUCCESS! Collected all {len(solver.treasures)} treasures!")
        print(f"Path length: {len(solution_path)}, Total cost: {total_cost:.2f}")
        print("Path coordinates:", solution_path)
    else:
        print("FAILED! Could not collect all treasures - no valid path found!")
    
    current_step = 0
    last_step_time = 0
    step_delay = 500
    auto_play = False
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
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
        
        if auto_play and solution_path and current_time - last_step_time > step_delay:
            if current_step < len(solution_path) - 1:
                current_step += 1
                last_step_time = current_time
            else:
                auto_play = False
        
        screen.fill(COLORS['background'])
        hex_grid.draw_grid(screen, solution_path, current_step if solution_path else None, solver.jump_moves)
        
        if solution_path:
            step_text = font.render(f"PATH: {current_step}/{len(solution_path) - 1}", True, COLORS['text'])
            # Use precomputed states for accurate cost
            current_cost = 0.0
            for i in range(1, current_step + 1):
                prev_grav, prev_speed = step_states[i-1]
                move_cost = solver.calculate_movement_cost(prev_grav, prev_speed)
                current_cost += move_cost
            
            cost_text = font.render(f"COST: {current_cost:.2f}", True, COLORS['text'])
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
            grav_text = font.render(f"GRAV: {step_states[current_step][0]}", True, COLORS['text'])
            speed_text = font.render(f"SPEED: {step_states[current_step][1]}", True, COLORS['text'])
            screen.blit(grav_text, (10, 125))
            screen.blit(speed_text, (10, 165))
        
        # Strategy info
        strategy_text = pygame.font.Font(None, 28).render("Strategy: A* Algorithm, Shortest path to Nearest Target (Treasure Priority)", True, COLORS['text'])
        screen.blit(strategy_text, (10, 90))
        
        instructions = [
            "SPACE: Auto-play/Pause",
            "UP/DOWN ARROWS: Step through path by path",
            "R: Reset to start"
        ]

        for i, instruction in enumerate(instructions):
            text = pygame.font.Font(None, 24).render(instruction, True, COLORS['text'])
            screen.blit(text, (10, WINDOW_HEIGHT - 100 + i * 25))
        
        legend_items = [
            ("Start (S)", COLORS['start']),
            ("Treasure (T)", COLORS['treasure']),
            ("Traps (T1-T4)", COLORS['trap']),
            ("Rewards (R1-R2)", COLORS['reward']),
            ("Obstacle (O)", COLORS['obstacle']),
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