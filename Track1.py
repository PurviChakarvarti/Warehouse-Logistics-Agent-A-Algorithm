"""
AI Express Hackathon - Track 1: Warehouse Logistics Agent
Unit 2 - Informed Search (A* with Manhattan Distance heuristic)

Scenario:
An autonomous forklift picks up a package and delivers it to a
designated loading bay in a grid warehouse containing static shelf
(obstacle) cells.

Run:
    python agent.py

Requires:
    pip install matplotlib numpy
"""

import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------------------------------------------
# 1. WAREHOUSE GRID DEFINITION
#    0 = free cell, 1 = shelf / obstacle
# ------------------------------------------------------------------
GRID = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
]

START = (0, 0)          # forklift pickup point (row, col)
GOAL = (9,9)            # designated loading bay (row, col)

ROWS = len(GRID)
COLS = len(GRID[0])


# ------------------------------------------------------------------
# 2. HEURISTIC: MANHATTAN DISTANCE  h(n) = |x1-x2| + |y1-y2|
# ------------------------------------------------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------------
# 3. A* SEARCH
# ------------------------------------------------------------------
def neighbors(node):
    r, c = node
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for nr, nc in candidates:
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] == 0:
            yield (nr, nc)


def a_star(start, goal):
    open_heap = [(manhattan(start, goal), 0, start)]  # (f, g, node)
    came_from = {}
    g_score = {start: 0}
    closed = set()
    expanded_order = []          # for visualization / logging
    expanded_count = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expanded_order.append(current)
        expanded_count += 1

        print(f"[EXPAND #{expanded_count}] node={current}  g={g}  f={f:.1f}")

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, expanded_order, g_score[goal], expanded_count

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + 1  # uniform step cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None, expanded_order, float("inf"), expanded_count


# ------------------------------------------------------------------
# 4. RUN SEARCH + LOG RESULTS
# ------------------------------------------------------------------
print("=" * 60)
print("AI Express Hackathon - Track 1: Warehouse Logistics Agent")
print("Algorithm: A* Search | Heuristic: Manhattan Distance")
print("=" * 60)
print(f"Start: {START}   Goal: {GOAL}")
print("-" * 60)

t0 = time.time()
path, expanded_order, path_cost, expanded_count = a_star(START, GOAL)
t1 = time.time()

print("-" * 60)
if path:
    print(f"PATH FOUND. Total Path Cost: {path_cost}")
    print(f"Total Nodes Expanded: {expanded_count}")
    print(f"Execution Time: {(t1 - t0)*1000:.2f} ms")
    print(f"Path: {path}")
else:
    print("NO PATH FOUND.")
print("=" * 60)


# ------------------------------------------------------------------
# 5. MATPLOTLIB ANIMATION
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
grid_np = np.array(GRID)

def draw_base():
    ax.clear()
    ax.imshow(grid_np, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(START[1], START[0], marker="s", markersize=16, color="green", label="Start (Pickup)")
    ax.plot(GOAL[1], GOAL[0], marker="*", markersize=20, color="red", label="Goal (Loading Bay)")
    ax.set_title("Warehouse Logistics Agent - A* Search", fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=9)

# Frame sequence: first show explored nodes, then animate the final path
explored_xy = [(n[1], n[0]) for n in expanded_order]
path_xy = [(n[1], n[0]) for n in path] if path else []

total_frames = len(explored_xy) + len(path_xy)

def update(frame):
    draw_base()
    if frame < len(explored_xy):
        xs = [p[0] for p in explored_xy[:frame + 1]]
        ys = [p[1] for p in explored_xy[:frame + 1]]
        ax.scatter(xs, ys, c="skyblue", s=60, alpha=0.6, label="Expanded")
        ax.text(0.02, 1.05, f"Expanding nodes... ({frame+1}/{len(explored_xy)})",
                transform=ax.transAxes, fontsize=10, color="blue")
    else:
        p_idx = frame - len(explored_xy)
        xs = [p[0] for p in explored_xy]
        ys = [p[1] for p in explored_xy]
        ax.scatter(xs, ys, c="skyblue", s=40, alpha=0.3)
        px = [p[0] for p in path_xy[:p_idx + 1]]
        py = [p[1] for p in path_xy[:p_idx + 1]]
        ax.plot(px, py, c="orange", linewidth=3, marker="o", markersize=6, label="Forklift Path")
        ax.text(0.02, 1.05, f"Forklift moving... step {p_idx+1}/{len(path_xy)} | "
                             f"Cost so far: {p_idx}",
                transform=ax.transAxes, fontsize=10, color="darkorange")
    return []

ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=120, repeat=False)

# Save as GIF for the submission video / README, and show live window
ani.save("forklift_astar.gif", writer="pillow", fps=8)
print("Saved animation to forklift_astar.gif")

plt.show()