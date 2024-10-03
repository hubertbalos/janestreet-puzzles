from typing import List, Tuple
from copy import deepcopy
from collections import defaultdict

class Solution:
    def __init__(self):
        self.grid_template = [
            ["A", "B", "B", "C", "C", "C"],
            ["A", "B", "B", "C", "C", "C"],
            ["A", "A", "B", "B", "C", "C"],
            ["A", "A", "B", "B", "C", "C"],
            ["A", "A", "A", "B", "B", "C"],
            ["A", "A", "A", "B", "B", "C"]]
        
        self.top_left = (0, 0)
        self.bottom_right = (5, 5)
        self.bottom_left = (5, 0)
        self.top_right = (0, 5)

        self.start = None
        self.end = None
        self.lower_limit = 0
        self.upper_limit = 5

        # ------ADJUSTABLE ------
        self.triplet_bound = 5
        self.dfs_depth_limit = 12
        # -----------------------

        self.target = 2024
        self.triplets = self.generate_triplets()
    
    def solve(self):
        """Solve problem for all triplets."""
        
        print("--------- RED PATH ---------")
        red_paths = defaultdict(list)
        self.start = self.bottom_left
        self.end = self.top_right
        total, i = len(self.triplets), 1

        for triplet in self.triplets:
            paths = self.solve_for_triplet(triplet)
            if paths:
                red_paths[triplet].append(min(paths, key=len))
            print(f"\r{i}/{total} Triplets explored", end="", flush=True)
            i += 1
        
        print("\n--------- BLUE PATH ---------")
        blue_paths = defaultdict(list)
        self.start = self.top_left
        self.end = self.bottom_right
        total, i = len(self.triplets), 1

        for triplet in self.triplets:
            paths = self.solve_for_triplet(triplet)
            if paths:
                blue_paths[triplet].append(min(paths, key=len))
            print(f"\r{i}/{total} Triplets explored", end="", flush=True)
            i += 1
        
        common_triplets = set(red_paths.keys()) & set(blue_paths.keys())
        if common_triplets:
            min_triplet = min(common_triplets, key=lambda triplet: sum(triplet))
            
            print(f"\n\nTriplet: {min_triplet}")
            print(f"Red path: {self.map_coordinates(red_paths[min_triplet])}")
            print(f"Blue path: {self.map_coordinates(blue_paths[min_triplet])}")

            triplet_str = ','.join(map(str, min_triplet))
            red_path_str = ','.join(self.map_coordinates(red_paths[min_triplet]))
            blue_path_str = ','.join(self.map_coordinates(blue_paths[min_triplet]))
            print("\n--------- PASTE ---------")
            print(f"{triplet_str},{red_path_str},{blue_path_str}")
        else:
            print("No common triplets.")

    def map_coordinates(self, paths):
        x_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f"}
        y_map = {0: "6", 1: "5", 2: "4", 3: "3", 4: "2", 5: "1"}
        mapped_path = []

        path = paths[0]
        for (y, x) in path:
            mapped_coordinate = x_map[x] + y_map[y]
            mapped_path.append(mapped_coordinate)

        return mapped_path
    
    def solve_for_triplet(self, triplet: Tuple[int, int, int]) -> List[List[Tuple[int, int]]]:
        """Solve the grid for a specific triplet and return all valid paths."""
        a, b, c = triplet
        grid = self.set_grid_with_triplet(a, b, c)  # Set grid locally

        all_paths = []
        start_position = self.start
        initial_score = grid[start_position[0]][start_position[1]]

        # Perform DFS for this particular grid setup, initialising visited set
        self.dfs(start_position, [start_position], initial_score, grid, all_paths, set([start_position]), 0)
        return all_paths
    
    def dfs(self, current_position: Tuple[int, int], current_path: List[Tuple[int, int]], current_score: int, grid: List[List[int]], all_paths: List[List[Tuple[int, int]]], visited: set, depth: int):
        """Performs a depth-first search to explore all move possibilities while avoiding revisiting positions."""
        if depth == self.dfs_depth_limit: # Exit early if dfs depth limit has been reached
            return
        if current_score > self.target:  # If the score exceeds the target, stop exploring this path
            return
        if current_position == self.end and current_score == self.target:  # If we reach the end position and meet the score
            all_paths.append(deepcopy(current_path))  # Store the valid path
            return

        possible_moves = self.get_possible_moves(current_position)
        
        for move in possible_moves:
            if move not in visited:  # Avoid revisiting positions already in this path
                # Add the move to visited and the current path
                visited.add(move)
                current_path.append(move)

                # Calculate the score for the next move
                new_score = self.compute_new_score(current_score, current_position, move, grid)
                
                # Recursively explore the next move
                self.dfs(move, current_path, new_score, grid, all_paths, visited, depth + 1)
                
                # Backtrack: remove the move from current path and visited set
                current_path.pop()
                visited.remove(move)

    def compute_new_score(self, current_score: int, current_position: Tuple[int, int], next_position: Tuple[int, int], grid: List[List[int]]) -> int:
        """Compute the new score based on the value at the next position."""
        current_value = grid[current_position[0]][current_position[1]]
        next_value = grid[next_position[0]][next_position[1]]
        
        # Update score logic: add if same value, multiply if different
        if current_value == next_value:
            return current_score + next_value
        else:
            return current_score * next_value
    
    def set_grid_with_triplet(self, a: int, b: int, c: int) -> List[List[int]]:
        """Set the grid with the provided triplet (a, b, c) and return the updated grid."""
        grid = deepcopy(self.grid_template)

        for row in range(self.upper_limit + 1):
            for column in range(self.upper_limit + 1):
                if grid[row][column] == "A":
                    grid[row][column] = a
                elif grid[row][column] == "B":
                    grid[row][column] = b
                elif grid[row][column] == "C":
                    grid[row][column] = c
                else:
                    raise ValueError("Unknown grid value")

        return grid

    def generate_triplets(self) -> List[Tuple[int, int, int]]:
        """Generate unique triplets whose sum is between 1 and 49."""
        triplets = []
        
        for a in range(1, self.triplet_bound + 1):
            for b in range(1, self.triplet_bound + 1):
                if b == a:
                    continue
                for c in range(1, self.triplet_bound + 1):
                    if c == a or c == b:
                        continue
                    if 1 <= a + b + c <= 49:
                        triplets.append((a, b, c))
        
        return triplets
    
    def get_possible_moves(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all possible moves from the current position."""
        possible_moves = []

        # UP
        position = [start[0] - 2, start[1]]
        if position[0] >= self.lower_limit:
            # UP -> RIGHT
            position[1] += 1
            if position[1] <= self.upper_limit:
                possible_moves.append(tuple(position))
            # UP -> LEFT
            position[1] -= 2
            if position[1] >= self.lower_limit:
                possible_moves.append(tuple(position))
        # RIGHT
        position = [start[0], start[1] + 2]
        if position[1] <= self.upper_limit:
            # RIGHT -> UP
            position[0] -= 1
            if position[0] >= self.lower_limit:
                possible_moves.append(tuple(position))
            # RIGHT -> DOWN
            position[0] += 2
            if position[0] <= self.upper_limit:
                possible_moves.append(tuple(position))
        # DOWN
        position = [start[0] + 2, start[1]]
        if position[0] <= self.upper_limit:
            # DOWN -> RIGHT
            position[1] += 1
            if position[1] <= self.upper_limit:
                possible_moves.append(tuple(position))
            # DOWN -> LEFT
            position[1] -= 2
            if position[1] >= self.lower_limit:
                possible_moves.append(tuple(position))
        # LEFT
        position = [start[0], start[1] - 2]
        if position[1] >= self.lower_limit:
            # LEFT -> UP
            position[0] -= 1
            if position[0] >= self.lower_limit:
                possible_moves.append(tuple(position))
            # LEFT -> DOWN
            position[0] += 2
            if position[0] <= self.upper_limit:
                possible_moves.append(tuple(position))

        return possible_moves

solution = Solution()
solution.solve()