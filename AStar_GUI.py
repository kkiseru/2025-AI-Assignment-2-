import pygame
import heapq
import math
import time
from math import inf, cos, sin, pi

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
COLORS = {
    'background': (20, 20, 30),
    'path': (255, 255, 255),
    'blocked': (40, 40, 50),
    'start': (0, 255, 0),
    'treasure': (255, 165, 0),    # Yellow like in reference image
    'obstacle': (128, 128, 128), # Gray like in reference image
    'trap': (147, 112, 219),     # Purple like in reference image
    'reward': (64, 224, 208),    # Turquoise/Teal like in reference image
    'current_path': (255, 255, 0),
    'visited_once': (100, 150, 255),     # Light blue for single visit
    'visited_multiple': (0, 100, 150),   # Dark blue for multiple visits
    'text': (255, 255, 255)
}

class HexGrid:
    def __init__(self, grid_data, hex_size=30):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        self.hex_size = hex_size
        
        # For flat-top hexagons (proper honeycomb orientation)
        # Width is 2 * size, height is sqrt(3) * size
        self.hex_width = hex_size * 2
        self.hex_height = hex_size * math.sqrt(3)
        
        # Reduced gaps between hexagons for tighter arrangement
        self.horizontal_spacing = self.hex_width * 0.85    # Standard hex spacing
        self.vertical_spacing = self.hex_height * 0.55     # Much smaller vertical gap
        
        # Calculate grid offset to center it
        total_width = (self.cols - 1) * self.horizontal_spacing + self.hex_width
        total_height = (self.rows - 1) * self.vertical_spacing + self.hex_height
        
        self.offset_x = (WINDOW_WIDTH - total_width) // 2
        self.offset_y = (WINDOW_HEIGHT - total_height) // 2
        
    def hex_to_pixel(self, row, col):
        """Convert hex grid coordinates to pixel coordinates for straight grid alignment"""
        x = self.offset_x + col * self.horizontal_spacing + self.hex_width // 2
        # No offset - all hexagons aligned in straight rows and columns
        y = self.offset_y + row * self.vertical_spacing + self.hex_height // 2
        return int(x), int(y)
    
    def draw_hexagon(self, surface, center_x, center_y, color, border_color=None):
        """Draw a flat-top hexagon (like honeycomb) at the given center position"""
        points = []
        for i in range(6):
            # Flat-top hexagon: start from right and go counter-clockwise
            # Angles: 0°, 60°, 120°, 180°, 240°, 300°
            angle = i * pi / 3
            x = center_x + self.hex_size * cos(angle)
            y = center_y + self.hex_size * sin(angle)
            points.append((x, y))
        
        # Draw filled hexagon
        pygame.draw.polygon(surface, color, points)
        
        # Draw border with slightly thicker line for better honeycomb effect
        border_col = border_color if border_color else (60, 60, 60)
        pygame.draw.polygon(surface, border_col, points, 1)
    
    def get_cell_color(self, cell_value):
        """Get color for a cell based on its value"""
        if cell_value == 1:
            return COLORS['path']
        elif cell_value == 0:
            return COLORS['blocked']
        elif cell_value == 'S':
            return COLORS['start']
        elif cell_value == 'T':
            return COLORS['treasure']
        elif cell_value == 'O':
            return COLORS['obstacle']
        elif cell_value in ['⊖', '⊕', '⊗', '⊘']:  # All traps same color
            return COLORS['trap']
        elif cell_value in ['⊞', '⊠']:  # All rewards same color
            return COLORS['reward']
        else:
            return COLORS['path']
    
    def draw_grid(self, surface, path=None, current_step=None):
        """Draw hexagons in straight grid alignment - skip blocked cells (0)"""
        # Count how many times each cell is visited in the path up to current step
        visit_count = {}
        current_pos = None
        
        if path and current_step is not None:
            # Count visits only up to current step
            for i in range(min(current_step + 1, len(path))):
                pos = path[i]
                visit_count[pos] = visit_count.get(pos, 0) + 1
                if i == current_step:
                    current_pos = pos
        
        for row in range(self.rows):
            for col in range(self.cols):
                cell_value = self.grid[row][col]
                
                # Skip blocked cells (0) - these should be empty spaces
                if cell_value == 0:
                    continue
                
                center_x, center_y = self.hex_to_pixel(row, col)
                
                # Get base color
                color = self.get_cell_color(cell_value)
                border_color = (100, 100, 100)
                
                # Check if this is the current position (always yellow regardless of visit count)
                if current_pos and (row, col) == current_pos:
                    # Current position - bright yellow
                    color = COLORS['current_path']
                    border_color = (255, 255, 255)
                # Check if this cell is part of the visited path (but not current position)
                elif path and (row, col) in visit_count:
                    visits = visit_count[(row, col)]
                    if visits > 1:
                        # Dark blue for multiple visits
                        color = COLORS['visited_multiple']
                        border_color = (0, 50, 100)
                    else:
                        # Light blue for single visit
                        color = COLORS['visited_once']
                        border_color = (50, 100, 200)
                
                # Draw hexagon only for valid cells
                self.draw_hexagon(surface, center_x, center_y, color, border_color)
                
                # Draw symbol/text for special cells with better Unicode support
                if cell_value in ['S', 'T', '⊖', '⊕', '⊗', '⊘', '⊞', '⊠']:
                    # Use a Unicode-compatible font
                    try:
                        font = pygame.font.Font(None, 28)
                    except:
                        font = pygame.font.SysFont('arial', 20)
                    
                    # Determine text color based on background
                    if current_pos and (row, col) == current_pos:
                        # Black text on yellow background for better visibility
                        text_color = (0, 0, 0)
                    elif (row, col) in visit_count:
                        # White text on blue backgrounds
                        text_color = (255, 255, 255)
                    else:
                        # Black text on original colored backgrounds, except for start
                        text_color = (0, 0, 0) if cell_value != 'S' else (255, 255, 255)
                    
                    # Handle special symbols with clearer representations
                    display_text = str(cell_value)
                    if cell_value == '⊖':
                        display_text = 'T1'  # Trap 1 - gravity trap
                    elif cell_value == '⊕':
                        display_text = 'T2'  # Trap 2 - speed trap
                    elif cell_value == '⊗':
                        display_text = 'T3'  # Trap 3 - teleport trap
                    elif cell_value == '⊘':
                        display_text = 'T4'  # Trap 4 - treasure removal trap
                    elif cell_value == '⊞':
                        display_text = 'R1'  # Reward 1 - gravity reward
                    elif cell_value == '⊠':
                        display_text = 'R2'  # Reward 2 - speed reward
                    
                    text = font.render(display_text, True, text_color)
                    text_rect = text.get_rect(center=(center_x, center_y))
                    surface.blit(text, text_rect)

class TreasureHuntAStar:
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0])
        
        # Find start and treasures
        self.start = None
        self.treasures = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell == 'S':
                    self.start = (i, j)
                elif cell == 'T':
                    self.treasures.append((i, j))
        
        self.treasures = tuple(sorted(self.treasures))
        self.directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
    def heuristic(self, row, col, remaining):
        """Admissible heuristic: Manhattan distance + MST"""
        if not remaining:
            return 0
        
        rem_list = list(remaining)
        dists_from_current = [abs(row - tr[0]) + abs(col - tr[1]) for tr in rem_list]
        min_dist = min(dists_from_current)
        
        mst_cost = 0
        if len(rem_list) > 1:
            visited = {rem_list[0]}
            not_visited = set(rem_list[1:])
            dist = {}
            for t1 in rem_list:
                for t2 in rem_list:
                    if t1 != t2:
                        dist[(t1, t2)] = abs(t1[0] - t2[0]) + abs(t1[1] - t2[1])
            
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
        """Solve the treasure hunt using A* algorithm"""
        start_state = (self.start[0], self.start[1], self.treasures, 0, 0)
        open_heap = [(self.heuristic(self.start[0], self.start[1], self.treasures), 0, start_state)]
        best_cost = {start_state: 0}
        parent = {start_state: None}
        
        goal_state = None
        visited_rewards = set()
        
        while open_heap:
            f, g, state = heapq.heappop(open_heap)
            row, col, remaining, grav_level, speed_level = state
            
            if not remaining and goal_state is None:
                goal_state = state
                break
            
            if g > best_cost.get(state, inf):
                continue
            
            for dr, dc in self.directions:
                nr, nc = row + dr, col + dc
                
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if self.grid[nr][nc] == 'O' or self.grid[nr][nc] == 0:
                    continue
                
                energy_cost = 2**grav_level
                time_cost = 2**speed_level
                move_cost = energy_cost + time_cost
                new_grav, new_speed = grav_level, speed_level
                new_remaining = remaining
                extra_cost = 0
                
                cell = self.grid[nr][nc]
                if cell == '⊖':
                    new_grav += 1
                elif cell == '⊕':
                    new_speed += 1
                elif cell == '⊗':
                    trap_coord = (nr, nc)
                    fr1, fc1 = nr + dr, nc + dc
                    if fr1 < 0 or fr1 >= self.rows or fc1 < 0 or fc1 >= self.cols or self.grid[fr1][fc1] == 'O':
                        continue
                    
                    fr2, fc2 = fr1 + dr, fc1 + dc
                    if fr2 < 0 or fr2 >= self.rows or fc2 < 0 or fc2 >= self.cols or self.grid[fr2][fc2] == 'O':
                        continue
                    
                    extra_cost = 2 * move_cost
                    intermediate_state = (trap_coord[0], trap_coord[1], remaining, grav_level, speed_level)
                    if intermediate_state not in best_cost:
                        best_cost[intermediate_state] = g + move_cost
                        parent[intermediate_state] = state
                    
                    nr, nc = fr2, fc2
                    cell = self.grid[nr][nc]
                    
                    if cell == 'T' and (nr, nc) in remaining:
                        rem_set = set(remaining)
                        rem_set.remove((nr, nc))
                        new_remaining = tuple(sorted(rem_set))
                    
                    new_state = (nr, nc, new_remaining, new_grav, new_speed)
                    new_g = g + move_cost + extra_cost
                    
                    if new_g < best_cost.get(new_state, inf):
                        best_cost[new_state] = new_g
                        parent[new_state] = intermediate_state
                        heapq.heappush(open_heap, (new_g + self.heuristic(nr, nc, new_remaining), new_g, new_state))
                        continue
                
                elif cell == '⊘':
                    if remaining:
                        continue
                elif cell == '⊞':
                    if (nr, nc) not in visited_rewards:
                        new_grav -= 1
                        visited_rewards.add((nr, nc))
                elif cell == '⊠':
                    if (nr, nc) not in visited_rewards:
                        new_speed -= 1
                        visited_rewards.add((nr, nc))
                
                if cell == 'T' and (nr, nc) in remaining:
                    rem_set = set(remaining)
                    rem_set.remove((nr, nc))
                    new_remaining = tuple(sorted(rem_set))
                
                new_g = g + move_cost + extra_cost
                new_state = (nr, nc, new_remaining, new_grav, new_speed)
                
                if new_g < best_cost.get(new_state, inf):
                    best_cost[new_state] = new_g
                    parent[new_state] = state
                    heapq.heappush(open_heap, (new_g + self.heuristic(nr, nc, new_remaining), new_g, new_state))
        
        # Reconstruct path
        path = []
        if goal_state:
            st, raw_path = goal_state, []
            while st:
                raw_path.append((st[0], st[1]))
                st = parent.get(st)
            raw_path.reverse()
            
            path = [raw_path[0]] + [p for i, p in enumerate(raw_path[1:], 1)
                                   if p != raw_path[i-1]]
            
            return path, best_cost[goal_state]
        
        return None, None

def main():
    # Grid data from your original code
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
    
    # Initialize display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A* Treasure Hunt - Hexagonal Grid (Straight Alignment)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Create hex grid and solver
    hex_grid = HexGrid(grid)
    solver = TreasureHuntAStar(grid)
    
    # Solve the puzzle
    print("Solving treasure hunt...")
    solution_path, total_cost = solver.solve()
    
    if solution_path:
        print(f"Solution found! Path length: {len(solution_path)}, Total cost: {total_cost}")
    else:
        print("No solution found!")
    
    # Animation variables
    current_step = 0
    last_step_time = 0
    step_delay = 500  # milliseconds between steps
    auto_play = False
    
    # Main game loop
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
        
        # Auto-play animation
        if auto_play and solution_path and current_time - last_step_time > step_delay:
            if current_step < len(solution_path) - 1:
                current_step += 1
                last_step_time = current_time
            else:
                auto_play = False
        
        # Clear screen
        screen.fill(COLORS['background'])
        
        # Draw grid
        hex_grid.draw_grid(screen, solution_path, current_step if solution_path else None)
        
        # Draw UI
        if solution_path:
            step_text = font.render(f"Step: {current_step + 1}/{len(solution_path)}", True, COLORS['text'])
            cost_text = font.render(f"Total Cost: {total_cost:.1f}", True, COLORS['text'])
            path_text = font.render("Light blue: visited once, Dark blue: visited multiple times", True, (100, 150, 255))
            screen.blit(step_text, (10, 10))
            screen.blit(cost_text, (10, 50))
            screen.blit(path_text, (10, 90))
        
        # Instructions
        instructions = [
            "SPACE: Auto-play/Pause",
            "LEFT/RIGHT: Step through path",
            "R: Reset to start"
        ]
        
        for i, instruction in enumerate(instructions):
            text = pygame.font.Font(None, 24).render(instruction, True, COLORS['text'])
            screen.blit(text, (10, WINDOW_HEIGHT - 100 + i * 25))
        
        # Legend with updated symbols
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