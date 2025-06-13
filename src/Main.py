import pygame
import heapq
import math
from math import inf, cos, sin, pi
from itertools import permutations, combinations

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


class TreasureHuntTSP:
    """
    Reward-Aware TSP optimization for treasure hunt problem.
    Pre-computes optimal route considering treasures and beneficial rewards.
    """
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        self.start = None
        self.treasures = []
        self.beneficial_rewards = []
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
                elif cell == '⊞':  # Gravity reducer - beneficial
                    self.beneficial_rewards.append(((i, j), 'gravity'))
                elif cell == '⊠':  # Speed reducer - beneficial
                    self.beneficial_rewards.append(((i, j), 'speed'))
        
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
        self.distance_cache = {}
        self.path_cache = {}
    
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
        if speed_level >= 0:
            step_multiplier = 2 ** speed_level
        else:
            step_multiplier = 1.0 / (2 ** abs(speed_level))
        
        return self.BASE_COST * (2 ** gravity_level) * step_multiplier
    
    def find_shortest_path(self, start_pos, end_pos, start_gravity=0, start_speed=0, collected_rewards=None):
        """Find shortest path between two points using A* with game mechanics"""
        if collected_rewards is None:
            collected_rewards = set()
        
        cache_key = (start_pos, end_pos, start_gravity, start_speed, frozenset(collected_rewards))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key], self.path_cache[cache_key]
        
        open_heap = [(0, 0, start_pos[0], start_pos[1], start_gravity, start_speed, frozenset(collected_rewards))]
        best_g_score = {}
        parent = {}
        
        start_state = (start_pos[0], start_pos[1], start_gravity, start_speed, frozenset(collected_rewards))
        best_g_score[start_state] = 0
        parent[start_state] = None
        
        while open_heap:
            f, g, row, col, grav_level, speed_level, used_rewards = heapq.heappop(open_heap)
            
            if (row, col) == end_pos:
                # Reconstruct path
                path = []
                state = (row, col, grav_level, speed_level, used_rewards)
                while state:
                    path.append((state[0], state[1]))
                    state = parent.get(state)
                path.reverse()
                
                self.distance_cache[cache_key] = g
                self.path_cache[cache_key] = path
                return g, path
            
            current_state = (row, col, grav_level, speed_level, used_rewards)
            if g > best_g_score.get(current_state, inf):
                continue
            
            valid_moves = self.get_valid_moves(row, col)
            
            for nr, nc, is_jump in valid_moves:
                new_grav, new_speed = grav_level, speed_level
                new_used_rewards = used_rewards
                
                cell = self.grid[nr][nc]
                
                # Handle game mechanics
                if cell == '⊖':
                    new_grav += 1
                elif cell == '⊕':
                    new_speed += 1
                elif cell == '⊘':
                    continue  # Skip paths through this trap for simplicity
                elif cell == '⊗':
                    continue  # Skip teleport traps for path computation
                elif cell == '⊞':
                    reward_key = (nr, nc)
                    if reward_key not in used_rewards:
                        new_grav = max(0, new_grav - 1)
                        new_used_rewards = used_rewards | {reward_key}
                elif cell == '⊠':
                    reward_key = (nr, nc)
                    if reward_key not in used_rewards:
                        new_speed = max(0, new_speed - 1)
                        new_used_rewards = used_rewards | {reward_key}
                
                move_cost = self.calculate_movement_cost(grav_level, speed_level)
                new_g = g + move_cost
                
                if is_jump:
                    self.jump_moves.add(((row, col), (nr, nc)))
                
                new_state = (nr, nc, new_grav, new_speed, new_used_rewards)
                
                if new_g < best_g_score.get(new_state, inf):
                    best_g_score[new_state] = new_g
                    parent[new_state] = current_state
                    
                    # Manhattan distance heuristic
                    h_score = abs(nr - end_pos[0]) + abs(nc - end_pos[1])
                    f_score = new_g + h_score
                    
                    heapq.heappush(open_heap, (f_score, new_g, nr, nc, new_grav, new_speed, new_used_rewards))
        
        return inf, []
    
    def calculate_reward_benefit(self, reward_pos, reward_type, current_gravity, current_speed):
        """Calculate the benefit of collecting a reward based on current state"""
        if reward_type == 'gravity':
            # Benefit increases with higher gravity levels
            return max(0, current_gravity) * 0.5
        elif reward_type == 'speed':
            # Benefit increases with higher speed levels
            return max(0, current_speed) * 0.3
        return 0
    
    def solve_tsp_with_rewards(self):
        """Solve TSP considering both treasures and beneficial rewards"""
        print("Computing optimal route with Reward-Aware TSP...")
        
        # All mandatory targets (treasures)
        mandatory_targets = self.treasures
        
        # All optional targets (beneficial rewards)
        optional_targets = [pos for pos, _ in self.beneficial_rewards]
        
        best_cost = inf
        best_route = None
        best_path = None
        
        # Try different combinations of optional rewards
        for r in range(len(optional_targets) + 1):
            for reward_combo in combinations(optional_targets, r):
                # Create full target list: treasures + selected rewards
                all_targets = list(mandatory_targets) + list(reward_combo)
                
                # Try all permutations of this target combination
                for perm in permutations(all_targets):
                    route_cost = 0
                    full_path = [self.start]
                    current_pos = self.start
                    current_gravity = 0
                    current_speed = 0
                    collected_rewards = set()
                    
                    # Calculate cost for this route
                    valid_route = True
                    for target in perm:
                        cost, path = self.find_shortest_path(
                            current_pos, target, current_gravity, current_speed, collected_rewards
                        )
                        
                        if cost == inf:
                            valid_route = False
                            break
                        
                        route_cost += cost
                        full_path.extend(path[1:])  # Skip first element to avoid duplication
                        current_pos = target
                        
                        # Update state based on what we collected
                        cell = self.grid[target[0]][target[1]]
                        if cell == '⊞':
                            current_gravity = max(0, current_gravity - 1)
                            collected_rewards.add(target)
                        elif cell == '⊠':
                            current_speed = max(0, current_speed - 1)
                            collected_rewards.add(target)
                        elif cell == '⊖':
                            current_gravity += 1
                        elif cell == '⊕':
                            current_speed += 1
                    
                    if valid_route and route_cost < best_cost:
                        best_cost = route_cost
                        best_route = perm
                        best_path = full_path
                        print(f"New best route found: cost={best_cost:.2f}, targets={len(perm)}")
        
        return best_path, best_cost, best_route
    
    def solve(self):
        """Main solve method using TSP optimization"""
        self.jump_moves.clear()
        
        # Use TSP optimization
        path, cost, route = self.solve_tsp_with_rewards()
        
        if path:
            print(f"Optimal route: {route}")
            return path, cost
        
        # Fallback to simple approach if TSP fails
        print("TSP optimization failed, using fallback...")
        return self.solve_simple()
    
    def solve_simple(self):
        """Simple fallback solution - just collect treasures in order"""
        path = [self.start]
        current_pos = self.start
        total_cost = 0
        
        for treasure in self.treasures:
            cost, segment = self.find_shortest_path(current_pos, treasure)
            if cost == inf:
                return None, None
            total_cost += cost
            path.extend(segment[1:])
            current_pos = treasure
        
        return path, total_cost


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
    pygame.display.set_caption("A* Treasure Hunt - Reward-Aware TSP Optimization")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    hex_grid = HexGrid(grid)
    solver = TreasureHuntTSP(grid)

    solution_path, total_cost = solver.solve()
    
    if solution_path:
        print(f"SUCCESS! Optimal route found!")
        print(f"Path length: {len(solution_path)}, Total cost: {total_cost:.2f}")
        print("Path coordinates:", solution_path)
    else:
        print("FAILED! Could not find a valid path!")
    
    current_step = 0
    last_step_time = 0
    step_delay = 300  # Faster animation for longer optimal paths
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
            cost_text = font.render(f"COST: {total_cost:.1f}", True, COLORS['text'])
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
        
        # Strategy info
        strategy_text = pygame.font.Font(None, 28).render("Strategy: Reward-Aware TSP Optimization", True, COLORS['text'])
        screen.blit(strategy_text, (10, 90))
        
        # Controls info
        controls_text = pygame.font.Font(None, 20).render("Controls: SPACE=Auto, LEFT/RIGHT=Step, R=Reset", True, COLORS['text'])
        screen.blit(controls_text, (10, 120))
        # Show control instructions
        instructions = [
            "SPACE: Auto-play/Pause",
            "LEFT/RIGHT: Step through path",
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