import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import heapq
from dataclasses import dataclass, field
from typing import Any, List, Set
import random

@dataclass(order=True)
class PuzzleState:
    """Represents a partial puzzle state, designed for a priority queue."""
    cost: float
    # field() is used to exclude non-comparable fields from the automatic __lt__ method
    placed_pieces: List[int] = field(compare=False)
    remaining_pieces: Set[int] = field(compare=False)

def load_and_split_image(mat_path, grid_size):
    """Load image from .mat file and split into puzzle pieces."""
    with open(mat_path, 'r') as f:
        lines = f.readlines()
    
    dims_line = lines[4].strip()
    dims = list(map(int, dims_line.split()))
    height, width = dims
    
    data_flat = [int(val) for line in lines[5:] for val in line.strip().split()]
    img_array = np.array(data_flat, dtype=np.uint8).reshape((height, width))
    img_array = np.stack([img_array] * 3, axis=-1)
    
    piece_h, piece_w = height // grid_size, width // grid_size
    pieces = [
        img_array[i*piece_h:(i+1)*piece_h, j*piece_w:(j+1)*piece_w, :]
        for i in range(grid_size) for j in range(grid_size)
    ]
    return pieces, piece_w, piece_h

def compute_edge_similarity(p1, p2, direction):
    """Compute similarity between edges (higher is better)."""
    if direction == 'right':
        edge1, edge2 = p1[:, -1, :], p2[:, 0, :]
    elif direction == 'bottom':
        edge1, edge2 = p1[-1, :, :], p2[0, :, :]
    else: return 0
    diff = edge1.astype(np.float32) - edge2.astype(np.float32)
    return -np.sqrt(np.sum(diff ** 2))

class BeamSearchSolver:
    """Solves a jigsaw puzzle using a beam search algorithm."""
    
    def __init__(self, pieces, grid_size, beam_width=150):
        self.pieces = pieces
        self.grid_size = grid_size
        self.num_pieces = grid_size * grid_size
        self.beam_width = beam_width
        self._precompute_similarities()

    def _precompute_similarities(self):
        """Precompute similarities between all piece edges for efficiency."""
        self.similarities = {}
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j: continue
                self.similarities[(i, j, 'right')] = compute_edge_similarity(self.pieces[i], self.pieces[j], 'right')
                self.similarities[(i, j, 'bottom')] = compute_edge_similarity(self.pieces[i], self.pieces[j], 'bottom')
    
    def solve(self):
        """Solve the puzzle using beam search."""
        initial_state = PuzzleState(
            cost=0.0,
            placed_pieces=[-1] * self.num_pieces,
            remaining_pieces=set(range(self.num_pieces))
        )
        
        beam = [initial_state]
        placement_order = list(range(self.num_pieces)) # Simple raster scan order
        
        for pos in placement_order:
            next_beam_candidates = []
            for state in beam:
                for piece_idx in state.remaining_pieces:
                    similarity = 0
                    row, col = pos // self.grid_size, pos % self.grid_size
                    
                    # Check left neighbor
                    if col > 0:
                        left_p = state.placed_pieces[pos - 1]
                        similarity += self.similarities.get((left_p, piece_idx, 'right'), 0)
                    # Check top neighbor
                    if row > 0:
                        top_p = state.placed_pieces[pos - self.grid_size]
                        similarity += self.similarities.get((top_p, piece_idx, 'bottom'), 0)
                        
                    new_placed = list(state.placed_pieces)
                    new_placed[pos] = piece_idx
                    new_remaining = state.remaining_pieces - {piece_idx}
                    
                    new_state = PuzzleState(
                        cost=state.cost - similarity, # Lower cost is better
                        placed_pieces=new_placed,
                        remaining_pieces=new_remaining
                    )
                    heapq.heappush(next_beam_candidates, new_state)

            # Keep only the top K candidates for the next beam
            beam = heapq.nsmallest(self.beam_width, next_beam_candidates)

        return min(beam, key=lambda s: s.cost).placed_pieces if beam else None

def calculate_solution_cost(solution, pieces, grid_size):
    """Calculates the total cost of a solved puzzle for display."""
    total_similarity = 0
    for pos, p_idx in enumerate(solution):
        row, col = pos // grid_size, pos % grid_size
        if col < grid_size - 1:
            total_similarity += compute_edge_similarity(pieces[p_idx], pieces[solution[pos + 1]], 'right')
        if row < grid_size - 1:
            total_similarity += compute_edge_similarity(pieces[p_idx], pieces[solution[pos + grid_size]], 'bottom')
    return -total_similarity

def reconstruct_image(solution, pieces, grid_size, piece_width, piece_height):
    """Reconstruct the full image from the solved piece list."""
    img = np.zeros((grid_size * piece_height, grid_size * piece_width, 3), dtype=np.uint8)
    for pos, piece_idx in enumerate(solution):
        row, col = pos // grid_size, pos % grid_size
        img[row*piece_height:(row+1)*piece_height, col*piece_width:(col+1)*piece_width] = pieces[piece_idx]
    return Image.fromarray(img)

def main(mat_path, grid_size=4, beam_width=150):
    """Main function to solve the jigsaw puzzle."""
    pieces, pw, ph = load_and_split_image(mat_path, grid_size)
    
    solver = BeamSearchSolver(pieces, grid_size, beam_width)
    solution = solver.solve()
    
    if solution:
        final_cost = calculate_solution_cost(solution, pieces, grid_size)
        scrambled_list = list(range(len(pieces)))
        random.shuffle(scrambled_list)
        
        scrambled_img = reconstruct_image(scrambled_list, pieces, grid_size, pw, ph)
        solved_img = reconstruct_image(solution, pieces, grid_size, pw, ph)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(scrambled_img); ax1.set_title('Scrambled'); ax1.axis('off')
        ax2.imshow(solved_img); ax2.set_title(f'Solved (Cost: {final_cost:.2f})'); ax2.axis('off')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main('scrambled_lena.mat', grid_size=4)