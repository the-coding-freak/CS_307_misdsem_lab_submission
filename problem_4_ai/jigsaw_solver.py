import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math

def load_and_split_image(mat_path, grid_size):
    """Load image from .mat file and split into pieces."""
    with open(mat_path, 'r') as f:
        lines = f.readlines()
    
    dims_line = lines[4].strip()
    dims = list(map(int, dims_line.split()))
    height, width = dims
    
    data_flat = [int(val) for line in lines[5:] for val in line.strip().split()]
    img_array = np.array(data_flat, dtype=np.uint8).reshape((height, width))
    img_array = np.stack([img_array] * 3, axis=-1)
    
    piece_h, piece_w = height // grid_size, width // grid_size
    pieces = []
    for i in range(grid_size):
        for j in range(grid_size):
            top, left = i * piece_h, j * piece_w
            pieces.append(img_array[top:top+piece_h, left:left+piece_w, :])
            
    return pieces, piece_w, piece_h, grid_size

def get_oriented_piece(piece, orientation):
    """Apply orientation to a piece (0=normal, 1=flipped)."""
    return piece if orientation == 0 else np.flipud(piece)

def compute_edge_dissimilarity(p1, p2, direction):
    """Compute L2 norm dissimilarity between adjacent edges."""
    if direction == 'right':
        edge1, edge2 = p1[:, -1, :], p2[:, 0, :]
    elif direction == 'bottom':
        edge1, edge2 = p1[-1, :, :], p2[0, :, :]
    else: return 0
    diff = edge1.astype(np.float32) - edge2.astype(np.float32)
    return np.sqrt(np.sum(diff ** 2))

def cost_function(state, pieces, grid_size):
    """Cost function with edge matching and corner bonuses."""
    total_cost = 0
    # Edge dissimilarity
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            p_idx, orient = state[idx]
            curr_p = get_oriented_piece(pieces[p_idx], orient)
            if j < grid_size - 1:
                r_p_idx, r_orient = state[idx + 1]
                right_p = get_oriented_piece(pieces[r_p_idx], r_orient)
                total_cost += compute_edge_dissimilarity(curr_p, right_p, 'right')
            if i < grid_size - 1:
                b_p_idx, b_orient = state[idx + grid_size]
                bottom_p = get_oriented_piece(pieces[b_p_idx], b_orient)
                total_cost += compute_edge_dissimilarity(curr_p, bottom_p, 'bottom')
    
    # Corner bonus
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            tl_s = state[i * grid_size + j]
            tr_s = state[i * grid_size + j + 1]
            bl_s = state[(i + 1) * grid_size + j]
            br_s = state[(i + 1) * grid_size + j + 1]
            tl_p = get_oriented_piece(pieces[tl_s[0]], tl_s[1])
            tr_p = get_oriented_piece(pieces[tr_s[0]], tr_s[1])
            bl_p = get_oriented_piece(pieces[bl_s[0]], bl_s[1])
            br_p = get_oriented_piece(pieces[br_s[0]], br_s[1])
            corners = np.array([tl_p[-1, -1], tr_p[-1, 0], bl_p[0, -1], br_p[0, 0]])
            total_cost += np.var(corners, axis=0).sum() * 0.1
            
    return total_cost

def generate_neighbor(state):
    """Generate neighbor states with various move strategies."""
    new_state = list(state)
    num_pieces = len(state)
    g = int(np.sqrt(num_pieces))
    rand = random.random()
    if rand < 0.4:  # Simple swap
        idx1, idx2 = random.sample(range(num_pieces), 2)
        new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
    elif rand < 0.6:  # Chain swap (3 pieces)
        indices = random.sample(range(num_pieces), 3)
        vals = [new_state[i] for i in indices]
        new_state[indices[0]], new_state[indices[1]], new_state[indices[2]] = vals[1], vals[2], vals[0]
    elif rand < 0.8 and g > 2:  # 2x2 block rotation
        i, j = random.randint(0, g - 2), random.randint(0, g - 2)
        tl, tr, bl, br = i*g+j, i*g+j+1, (i+1)*g+j, (i+1)*g+j+1
        new_state[tl], new_state[tr], new_state[bl], new_state[br] = new_state[bl], new_state[tl], new_state[br], new_state[tr]
    elif rand < 0.95:  # Multiple swaps
        for _ in range(random.randint(2, 4)):
            idx1, idx2 = random.sample(range(num_pieces), 2)
            new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
    else:  # Flip orientation
        idx = random.randint(0, num_pieces - 1)
        new_state[idx] = (new_state[idx][0], 1 - new_state[idx][1])
    return new_state

def adaptive_simulated_annealing(initial_state, pieces, g, max_iter=10000):
    """Enhanced simulated annealing solver."""
    curr_s = list(initial_state)
    curr_c = cost_function(curr_s, pieces, g)
    
    sample_costs = [abs(cost_function(generate_neighbor(curr_s), pieces, g) - curr_c) for _ in range(50)]
    temp = np.mean(sample_costs) * 2
    
    best_s, best_c = list(curr_s), curr_c
    cooling_rate, min_temp, stuck_counter, last_improvement = 0.995, 0.001, 0, 0
    
    for i in range(max_iter):
        if i - last_improvement > 500 and stuck_counter < 3:
            temp, stuck_counter, last_improvement = np.mean(sample_costs) * 2 * 0.5, stuck_counter + 1, i

        new_s = generate_neighbor(curr_s)
        new_c = cost_function(new_s, pieces, g)
        delta = new_c - curr_c
        
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            curr_s, curr_c = new_s, new_c
            if curr_c < best_c:
                best_s, best_c, last_improvement = list(curr_s), curr_c, i
        
        temp = max(min_temp, temp * cooling_rate)
        if best_c < 1.0: break
    return best_s

def smart_initialization(pieces, g):
    """Create a smarter initial state via greedy placement."""
    num_pieces = len(pieces)
    used, state = [False] * num_pieces, []
    
    first_idx = random.randint(0, num_pieces - 1)
    state.append((first_idx, 0))
    used[first_idx] = True
    
    for pos in range(1, num_pieces):
        best_p, best_c = None, float('inf')
        for p_idx in range(num_pieces):
            if used[p_idx]: continue
            cost = 0
            row, col = pos // g, pos % g
            if col > 0: cost += compute_edge_dissimilarity(pieces[state[pos-1][0]], pieces[p_idx], 'right')
            if row > 0: cost += compute_edge_dissimilarity(pieces[state[pos-g][0]], pieces[p_idx], 'bottom')
            if cost < best_c: best_c, best_p = cost, p_idx
        state.append((best_p, 0))
        used[best_p] = True
    return state

def multi_start_solve(pieces, g, num_starts=3):
    """Run solver multiple times to avoid local minima."""
    best_overall_s, best_overall_c = None, float('inf')
    for start in range(num_starts):
        initial_state = smart_initialization(pieces, g) if start > 0 else [(i,0) for i in range(g*g)]
        if start == 0: random.shuffle(initial_state)
        
        solved_s = adaptive_simulated_annealing(initial_state, pieces, g)
        final_c = cost_function(solved_s, pieces, g)
        
        if final_c < best_overall_c:
            best_overall_c, best_overall_s = final_c, solved_s
    return best_overall_s

def reconstruct_image(state, pieces, g, pw, ph):
    """Reconstruct the full image from puzzle state."""
    img = np.zeros((g * ph, g * pw, 3), dtype=np.uint8)
    for i in range(g):
        for j in range(g):
            p_idx, orient = state[i * g + j]
            img[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = get_oriented_piece(pieces[p_idx], orient)
    return Image.fromarray(img)

def main(mat_path, grid_size=4):
    """Main function to solve the jigsaw puzzle."""
    pieces, pw, ph, g = load_and_split_image(mat_path, grid_size)
    
    scrambled_state = [(i, 0) for i in range(g*g)]
    random.shuffle(scrambled_state)
    
    solved_state = multi_start_solve(pieces, g)
    
    final_cost = cost_function(solved_state, pieces, g)
    
    initial_img = reconstruct_image(scrambled_state, pieces, g, pw, ph)
    final_img = reconstruct_image(solved_state, pieces, g, pw, ph)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(initial_img); ax1.set_title('Scrambled'); ax1.axis('off')
    ax2.imshow(final_img); ax2.set_title(f'Solved (Cost: {final_cost:.2f})'); ax2.axis('off')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main('scrambled_lena.mat', grid_size=4)