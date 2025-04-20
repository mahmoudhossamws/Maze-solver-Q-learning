import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class CustomMazeEnv(gym.Env):
    def __init__(self):
        super(CustomMazeEnv, self).__init__()

        # Define maze dimensions (20x20)
        self.n_rows = 20
        self.n_cols = 20

        # Define the action space (4 possible actions: up, down, left, right)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right

        # Define the observation space (state space as a flattened 20x20 grid)
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)

        # Initialize the maze (custom 20x20 maze)
        self.maze = self.create_maze()

        # Define the initial position of the agent
        self.agent_pos = (0, 0)

        # Goal position
        self.goal_pos = (19, 19)

    def create_maze(self):

        maze = np.array([
            [0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
            [-1, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0],
            [-1, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0, -1, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0],
            [-1, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 1],
        ])
        return maze

    def reset(self):
        # Reset the agent's position to the start and return the initial state
        self.agent_pos = (0, 0)  # Starting at position (1, 1)
        return self.get_state()

    def get_state(self):
        # Convert the 2D position of the agent to a 1D index for the state space
        return self.agent_pos[0] * self.n_cols + self.agent_pos[1]

    def step(self, action):
        # Define the movement actions
        if action == 0:  # Up
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 1:  # Down
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 2:  # Left
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 3:  # Right
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

        if not (0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols):
            return self.get_state(), -5, False, {}

        # Check if the new position is  not a wall
        if self.maze[new_pos[0], new_pos[1]] != -1:  # Check if it's not a wall
            self.agent_pos = new_pos  # Update agent's position

        # Check if the agent has reached the goal
        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10, True, {}  # Goal reached, reward 10

        # Default step (if not goal or wall)
        return self.get_state(), -1, False, {}  # Step, negative reward

    def render(self):
        # Optional: Render the maze for visualization
        for i in range(self.n_rows):
            row = ""
            for j in range(self.n_cols):
                if (i, j) == self.agent_pos:
                    row += "A "  # 'A' represents the agent's position
                elif self.maze[i, j] == -1:
                    row += "# "  # '#' represents a wall
                elif self.maze[i, j] == 1:
                    row += "G "  # 'G' represents the goal
                else:
                    row += ". "  # '.' represents a free space
            print(row)
        print()  # New line after rendering the maze


env = CustomMazeEnv()
env.render()

Q_table = np.zeros((env.n_rows * env.n_cols, 4))

epochs = 500
max_steps = 1000

learningRate = 0.1
gamma = 0.99

epsilon = 1
state = env.reset()
for x in range(epochs):
    state = env.reset()
    for t in range(max_steps):
        if np.random.uniform() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q_table[state, :])

        next_state, reward, done, _ = env.step(action)

        Q_table[state, action] = Q_table[state, action] + learningRate * (
                reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action])
        state = next_state

        if done:
            break
    epsilon -= 0.00001
print(Q_table)


def getstate(i, j):
    return i * env.n_cols + j


path = []
current = 0
solved=True
count = 0
while current != len(Q_table) - 1:
    path.append(current)
    action = np.argmax(Q_table[current, :])

    if action == 0:  # Up
        current -= env.n_cols
    elif action == 1:  # Down
        current += env.n_cols
    elif action == 2:  # Left
        current -= 1
    elif action == 3:  # Right
        current += 1

    count += 1
    if count > 140:
        print("not solvable")
        solved=False
        break

if solved:
    print("solved:", current)
    path.append(current)
print(path)


def plot_path_with_walls(path, maze, rows, cols):
    plt.figure(figsize=(10, 10))

    # Plot maze grid
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == -1:
                plt.fill_between([j, j + 1], i, i + 1, color="black")  # Wall
            elif maze[i][j] == 1:
                plt.fill_between([j, j + 1], i, i + 1, color="green")  # Goal

    # Plot path
    wall_hit = False
    for idx in range(len(path) - 1):
        r1, c1 = divmod(path[idx], cols)
        r2, c2 = divmod(path[idx + 1], cols)

        # Red line if the current step is on a wall, else blue
        color = "red" if maze[r1][c1] == -1 else "blue"
        if maze[r1][c1] == -1:
            wall_hit = True
        plt.arrow(c1 + 0.5, r1 + 0.5, c2 - c1, r2 - r1,
                  head_width=0.2, length_includes_head=True, color=color)

    # Start position
    r0, c0 = divmod(path[0], cols)
    plt.plot(c0 + 0.5, r0 + 0.5, "go", markersize=10, label="Start")

    # End position
    r_end, c_end = divmod(path[-1], cols)
    plt.plot(c_end + 0.5, r_end + 0.5, "ro", markersize=10, label="End")

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xticks(range(cols + 1))
    plt.yticks(range(rows + 1))
    plt.legend()
    plt.title("Path Visualization (Red if Wall Hit)")
    plt.show()

    if wall_hit:
        print("⚠️ Warning: The path includes steps through walls!")
    else:
        print("✅ Path is valid and avoids walls.")


# Call the plot function
plot_path_with_walls(path, env.maze, env.n_rows, env.n_cols)