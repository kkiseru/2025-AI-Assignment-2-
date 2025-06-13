import pygame
import heapq
import math
from math import inf, cos, sin, pi
from functools import lru_cache
from collections import namedtuple

# Constants and Configuration
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
FPS = 60
HEX_SIZE = 30

# Data Structures
State = namedtuple('State', ['pos', 'treasures', 'grav', 'speed', 'used_grav', 'used_speed'])
Move = namedtuple('Move', ['dest', 'is_jump'])

# Color Palette
COLORS = {
    'background': (20, 20, 30),
    'path': (255, 255, 255),
    'blocked': (40, 40, 50),
    'start': (0, 255, 0),
    'treasure': (255, 165, 0),
    'obstacle': (128, 128, 128),
    'trap': (147, 112, 219),
    'reward': (64, 224, 208),
    'current_path': (255, 0, 0),
    'visited_once': (100, 150, 255),
    'visited_multiple': (0, 100, 150),
    'jump_path': (255, 100, 0),
    'text': (255, 255, 255),
    'teleport': (255, 0, 255)
}

class HexGrid:
    def __init__(self, grid_data, hex_size=HEX_SIZE):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.hex_size = hex_size
        self.hex_width = hex_size * 2
        self.hex_height = hex_size * math.sqrt(3)
        self.horizontal_spacing = self.hex_width * 0.85
        self.vertical_spacing = self.hex_height * 0.55
        
        # Precompute grid dimensions
        total_width = (self.cols - 1) * self.horizontal_spacing + self.hex_width
        total_height = (self.rows - 1) * self.vertical_spacing + self.hex_height
        self.offset_x = (WINDOW_WIDTH - total_width) // 2
        self.offset_y = (WINDOW_HEIGHT - total_height) // 2
        
        # Symbol mapping for display
        self.symbol_labels = {
            '⊖': 'T1', '⊕': 'T2', '⊗': 'T3', '⊘': 'T4',
            '⊞': 'R1', '⊠': 'R2'
        }

    def hex_to_pixel(self, row, col):
        x = self.offset_x + col * self.horizontal_spacing + self.hex_width // 2
        y = self.offset_y + row * self.vertical_spacing + self.hex_height // 2
        return int(x), int(y)

    def draw_hexagon(self, surface, center_x, center_y, color, border_color=None):
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
        if cell_value in ['⊖', '⊕', '⊗', '⊘']:
            return COLORS['trap']
        elif cell_value in ['⊞', '⊠']:
            return COLORS['reward']
        
        color_map = {
            1: COLORS['path'],
            0: COLORS['blocked'],
            'S': COLORS['start'],
            'T': COLORS['treasure'],
            'O': COLORS['obstacle'],
            '⊗': COLORS['teleport']
        }
        return color_map.get(cell_value, COLORS['path'])

    def draw_grid(self, surface, path=None, current_step=None, special_moves=None):
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
                
                # Highlighting logic
                if current_pos and (row, col) == current_pos:
                    color = COLORS['current_path']
                    border_color = (255, 255, 255)
                elif path and (row, col) in visit_count:
                    visits = visit_count[(row, col)]
                    is_jump = special_moves and any(
                        (src, (row, col)) in special_moves['jumps'] 
                        for src in [path[i] for i in range(min(current_step + 1, len(path) - 1))]
                    )
                    is_teleport = special_moves and (row, col) in special_moves['teleports']
                    
                    if visits > 1:
                        color = COLORS['visited_multiple']
                        border_color = (0, 50, 100)
                    else:
                        color = COLORS['visited_once']
                        border_color = (50, 100, 200)
                    
                    if is_jump:
                        color = COLORS['jump_path']
                    elif is_teleport:
                        color = COLORS['teleport']
                
                self.draw_hexagon(surface, center_x, center_y, color, border_color)
                
                # Draw cell labels
                if cell_value in ['S', 'T', '⊖', '⊕', '⊗', '⊘', '⊞', '⊠']:
                    try:
                        font = pygame.font.Font(None, 28)
                    except:
                        font = pygame.font.SysFont('arial', 20)
                    
                    text_color = (0, 0, 0) if color in [COLORS['current_path'], COLORS['jump_path']] else (255, 255, 255)
                    display_text = self.symbol_labels.get(cell_value, str(cell_value))
                    
                    text = font.render(display_text, True, text_color)
                    text_rect = text.get_rect(center=(center_x, center_y))
                    surface.blit(text, text_rect)

class TreasureHuntSolver:
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        # Precompute important positions
        self.start = None
        self.treasures = []
        self.rewards = {'grav': [], 'speed': []}
        
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
                elif cell == '⊞':
                    self.rewards['grav'].append((i, j))
                elif cell == '⊠':
                    self.rewards['speed'].append((i, j))
        
        self.treasures = tuple(sorted(self.treasures))
        self.total_rewards = {
            'grav': len(self.rewards['grav']),
            'speed': len(self.rewards['speed'])
        }
        
        # Movement directions (flat-top hex grid)
        self.directions = [
            (-1, 0), (1, 0),           # Up, Down
            (-1, -1), (-1, 1),         # Up-Left, Up-Right
            (1, -1), (1, 1)            # Down-Left, Down-Right
        ]
        
        # Cost parameters
        self.BASE_COST = 1.0
        self.MIN_COST = 0.125  # Minimum possible cost per step

    def get_valid_moves(self, pos):
        moves = []
        row, col = pos
        
        for dr, dc in self.directions:
            nr, nc = row + dr, col + dc
            
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
                
            cell_value = self.grid[nr][nc]
            
            # Vertical movement (up/down) with jump capability
            if (dr, dc) in [(-1, 0), (1, 0)]:
                if cell_value == 0:  # Jump over blocked cell
                    jump_r, jump_c = nr + dr, nc + dc
                    if (0 <= jump_r < self.rows and 0 <= jump_c < self.cols and
                        self.grid[jump_r][jump_c] not in ['O', 0]):
                        moves.append(Move((jump_r, jump_c), True))
                elif cell_value not in ['O', 0]:
                    moves.append(Move((nr, nc), False))
            # Diagonal movement (standard)
            elif cell_value not in ['O', 0]:
                moves.append(Move((nr, nc), False))
                
        return moves

    def calculate_cost(self, grav_level, speed_level):
        grav_mult = max(self.MIN_COST, 2 ** grav_level)
        speed_mult = max(self.MIN_COST, 2 ** speed_level)
        return self.BASE_COST * grav_mult * speed_mult

    @lru_cache(maxsize=None)
    def mst_cost(self, treasures):
        if len(treasures) <= 1:
            return 0
            
        remaining = set(treasures)
        visited = {remaining.pop()}
        total_cost = 0
        
        while remaining:
            min_dist = min(
                abs(x1 - x2) + abs(y1 - y2)
                for (x1, y1) in visited
                for (x2, y2) in remaining
            )
            total_cost += min_dist
            
            # Find and add the closest treasure
            for (x1, y1) in visited:
                for (x2, y2) in remaining:
                    if abs(x1 - x2) + abs(y1 - y2) == min_dist:
                        visited.add((x2, y2))
                        remaining.remove((x2, y2))
                        break
                else:
                    continue
                break
                
        return total_cost

    def heuristic(self, state):
        remaining = state.treasures
        if not remaining:
            return 0
            
        # Minimum distance to nearest treasure
        pos_x, pos_y = state.pos
        min_dist = min(abs(pos_x - x) + abs(pos_y - y) for (x, y) in remaining)
        
        # Minimum possible cost per step
        min_grav = state.grav - (self.total_rewards['grav'] - state.used_grav)
        min_speed = state.speed - (self.total_rewards['speed'] - state.used_speed)
        min_step_cost = self.calculate_cost(min_grav, min_speed)
        
        # MST cost for remaining treasures
        mst_cost = self.mst_cost(remaining)
        
        return (min_dist + mst_cost) * min_step_cost

    def solve(self):
        initial_state = State(
            pos=self.start,
            treasures=self.treasures,
            grav=0,
            speed=0,
            used_grav=0,
            used_speed=0
        )
        
        open_set = [(self.heuristic(initial_state), 0, initial_state)]
        g_scores = {initial_state: 0}
        parents = {initial_state: None}
        
        # Track special moves for visualization
        special_moves = {
            'jumps': set(),
            'teleports': set()
        }
        
        while open_set:
            _, g, current = heapq.heappop(open_set)
            
            if not current.treasures:
                path = []
                while current:
                    path.append(current.pos)
                    current = parents.get(current)
                return path[::-1], g, special_moves
                
            if g > g_scores.get(current, inf):
                continue
                
            for move in self.get_valid_moves(current.pos):
                new_pos = move.dest
                new_grav = current.grav
                new_speed = current.speed
                new_used_grav = current.used_grav
                new_used_speed = current.used_speed
                new_treasures = current.treasures
                
                cell = self.grid[new_pos[0]][new_pos[1]]
                
                # Handle traps and rewards
                if cell == '⊖':
                    new_grav += 1
                elif cell == '⊕':
                    new_speed += 1
                elif cell == '⊞' and new_used_grav < self.total_rewards['grav']:
                    new_grav -= 1
                    new_used_grav += 1
                elif cell == '⊠' and new_used_speed < self.total_rewards['speed']:
                    new_speed -= 1
                    new_used_speed += 1
                elif cell == 'T' and new_pos in new_treasures:
                    new_treasures = tuple(t for t in new_treasures if t != new_pos)
                
                # Handle teleport - move back to grandparent position
                if cell == '⊗' and current in parents and parents[current] in parents:
                    grandparent = parents[parents[current]]
                    new_pos = grandparent.pos
                    special_moves['teleports'].add(new_pos)
                
                # Calculate movement cost with current effects
                move_cost = self.calculate_cost(current.grav, current.speed)
                
                # Record jump moves for visualization
                if move.is_jump:
                    special_moves['jumps'].add((current.pos, new_pos))
                
                # Create new state
                new_state = State(
                    pos=new_pos,
                    treasures=new_treasures,
                    grav=new_grav,
                    speed=new_speed,
                    used_grav=new_used_grav,
                    used_speed=new_used_speed
                )
                
                new_g = g + move_cost
                
                if new_g < g_scores.get(new_state, inf):
                    g_scores[new_state] = new_g
                    parents[new_state] = current
                    heapq.heappush(
                        open_set,
                        (new_g + self.heuristic(new_state), new_g, new_state)
                    )
        
        return None, None, None  # No solution found

def main():
    # Sample grid (same as original)
    grid = [
        [   0,        1,        0,        1,        0,        1,        0,        1,        0,        1   ],
        [  'S',       0,        1,        0,       '⊞',      0,        1,        0,        1,        0   ],
        [   0,       '⊕',      0,       '⊘',      0,        1,        0,        1,        0,        1   ],
        [   1,        0,        1,        0,       'T',       0,       '⊗',      0,       'O',       0   ],
        [   0,        1,        0,        1,        0,        1,        0,       '⊠',      0,        1   ],
        [   1,        0,       'O',       0,       'O',       0,        1,        0,       '⊖',      0   ],
        [   0,       '⊞',      0,       'O',       0,       '⊗',      0,       'T',       0,       'T'  ],
        [  'O',       0,        1,        0,        1,        0,       'O',       0,        1,        0   ],
        [   0,        1,        0,       'T',       0,        1,        0,       'O',       0,        1   ],
        [   1,        0,       '⊕',      0,       'O',       0,       'O',       0,        1,        0   ],
        [   0,        1,        0,        1,        0,       '⊠',      0,        1,        0,        1   ],
        [   1,        0,        1,        0,        1,        0,        1,        0,        1,        0   ],
    ]
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Optimized A* Treasure Hunt")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Create solver and visualization
    hex_grid = HexGrid(grid)
    solver = TreasureHuntSolver(grid)
    
    # Solve the problem
    solution_path, total_cost, special_moves = solver.solve()
    
    # Print results
    if solution_path:
        print(f"Solution found! Collected {len(solver.treasures)} treasures.")
        print(f"Path length: {len(solution_path)}, Total cost: {total_cost}")
    else:
        print("No solution found!")
    
    # Animation controls
    current_step = 0
    last_step_time = 0
    step_delay = 500  # ms
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
        
        # Render
        screen.fill(COLORS['background'])
        hex_grid.draw_grid(screen, solution_path, current_step if solution_path else None, special_moves)
        
        # Display info
        if solution_path:
            step_text = font.render(f"Step: {current_step}/{len(solution_path) - 1}", True, COLORS['text'])
            
            # Calculate current cost
            current_cost = 0.0
            grav, speed = 0, 0
            used_grav, used_speed = 0, 0
            
            for i in range(1, current_step + 1):
                r, c = solution_path[i]
                cell = grid[r][c]
                
                # Calculate cost with current effects
                current_cost += solver.calculate_cost(grav, speed)
                
                # Update effects
                if cell == '⊖': grav += 1
                elif cell == '⊕': speed += 1
                elif cell == '⊞' and used_grav < solver.total_rewards['grav']:
                    grav -= 1
                    used_grav += 1
                elif cell == '⊠' and used_speed < solver.total_rewards['speed']:
                    speed -= 1
                    used_speed += 1
            
            cost_text = font.render(f"Cost: {current_cost}", True, COLORS['text'])
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
        
        # Legend
        legend_items = [
            ("Start (S)", COLORS['start']),
            ("Treasure (T)", COLORS['treasure']),
            ("Traps (T1-T4)", COLORS['trap']),
            ("Rewards (R1-R2)", COLORS['reward']),
            ("Obstacle (O)", COLORS['obstacle']),
            ("Teleport (T3)", COLORS['teleport']),
            ("Jump Path", COLORS['jump_path'])
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