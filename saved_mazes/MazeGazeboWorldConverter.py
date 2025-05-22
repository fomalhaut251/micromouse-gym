import sys
import os
import math

def generate_sdf_box(name, pose, size):
    """
    Generate the SDF definition for a simple box model.

    Args:
        name (str): The name of the model.
        pose (list): [x, y, z, roll, pitch, yaw] pose of the model.
        size (list): [length, width, height] size of the box.

    Returns:
        str: The SDF XML snippet for the box model.
    """
    pose_str = f"{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]}"
    size_str = f"{size[0]} {size[1]} {size[2]}"

    sdf = f"""
      <model name="{name}">
        <static>true</static> <pose>{pose_str}</pose>
        <link name="link">
          <collision name="collision">
            <geometry>
              <box>
                <size>{size_str}</size>
              </box>
            </geometry>
          </collision>
          <visual name="visual">
            <geometry>
              <box>
                <size>{size_str}</size>
              </box>
            </geometry>
            <material>
               <script>
                 <uri>file://media/materials/scripts/gazebo.material</uri>
                 <name>Gazebo/Grey</name> </script>
            </material>
          </visual>
        </link>
      </model>
    """
    return sdf

def generate_world_sdf(maze_data, unit=1.0, wall_height=0.5, wall_thickness=0.1):
    """
    Generate the Gazebo World SDF content from maze data.

    Args:
        maze_data (list of list): The 2D list of characters representing the maze.
        unit (float): The distance between the centers of adjacent cells/pillars.
        wall_height (float): The height of the walls.
        wall_thickness (float): The thickness of the walls.

    Returns:
        str: The complete Gazebo World SDF XML content.
    """
    rows = len(maze_data) # Expected 33
    cols = len(maze_data[0]) # Expected 33
    walls_sdf = ""
    wall_count = 0

    # The maze is 16x16 cells. The grid is 33x33.
    # Grid indices i, j (0 to 32) map to maze cell/pillar logic.
    # Pillars are at i, j where i%2==0 and j%2==0. These correspond to maze cell indices (i/2, j/2) for 0-indexed 16x16 grid.
    # Horizontal walls are at i, j where i%2==0 and j%2!=0. Between pillar (i/2, (j-1)/2) and (i/2, (j+1)/2).
    # Vertical walls are at i, j where i%2!=0 and j%2==0. Between pillar ((i-1)/2, j/2) and ((i+1)/2, j/2).

    # Calculate offset to center the maze around (0,0) in the Gazebo world.
    # The maze spans roughly 16 * unit in x and 16 * unit in y.
    offset_x = (16 / 2.0) * unit
    offset_y = (16 / 2.0) * unit

    # Iterate through the grid to find walls
    for i in range(rows):
        for j in range(cols):
            if maze_data[i][j] == '1':
                # Check if it's a wall location based on the rules
                if i % 2 == 0 and j % 2 != 0: # Even row, odd column -> Horizontal wall
                    # Wall is between pillar (i/2, (j-1)/2) and (i/2, (j+1)/2)
                    # World X: Center between j-1 and j+1 columns, adjusted for offset
                    # World Y: Same row as pillar (i/2), adjusted for offset
                    pose_x = ((j - 1) / 2.0 * unit + unit / 2.0) - offset_x # Simplified: (j/2 * unit) - offset_x
                    pose_y = -(i / 2.0 * unit) + offset_y # Simplified: (offset_y - i/2 * unit)
                    pose_z = wall_height / 2.0
                    pose_rpy = [0, 0, 0] # No rotation for horizontal wall

                    walls_sdf += generate_sdf_box(
                        f"wall_{wall_count}",
                        [pose_x, pose_y, pose_z, pose_rpy[0], pose_rpy[1], pose_rpy[2]],
                        [unit, wall_thickness, wall_height] # Box size: length (X), width (Y), height (Z)
                    )
                    wall_count += 1

                elif i % 2 != 0 and j % 2 == 0: # Odd row, even column -> Vertical wall
                    # Wall is between pillar ((i-1)/2, j/2) and ((i+1)/2, j/2)
                    # World X: Same column as pillar (j/2), adjusted for offset
                    # World Y: Center between i-1 and i+1 rows, adjusted for offset
                    pose_x = (j / 2.0 * unit) - offset_x
                    pose_y = -((i - 1) / 2.0 * unit + unit / 2.0) + offset_y # Simplified: (offset_y - i/2 * unit)
                    pose_z = wall_height / 2.0
                    pose_rpy = [0, 0, math.pi / 2.0] # Rotate 90 degrees (pi/2) around Z for vertical wall

                    walls_sdf += generate_sdf_box(
                        f"wall_{wall_count}",
                        [pose_x, pose_y, pose_z, pose_rpy[0], pose_rpy[1], pose_rpy[2]],
                        [unit, wall_thickness, wall_height] # Box size: length (X), width (Y), height (Z)
                    )
                    wall_count += 1
                # Ignore pillars (odd row, odd column) and cell centers (even row, even column)

    # Construct the complete SDF World XML
    sdf_content = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="maze_world">

    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <physics name="default_physics" default="0" type="ode">
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <rms_error>0.0</rms_error>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
        </constraints>
      </ode>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
    </physics>

    <gravity>0 0 -9.8</gravity>

    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>

    {walls_sdf}

  </world>
</sdf>
"""
    return sdf_content

def main():
    # Define input and output directories
    input_dir = "mazes_text"
    output_dir = "gazebo_world"

    # --- 硬编码输入和输出文件名 ---
    # 请在这里修改为你实际的迷宫文件名和想要生成的world文件名
    input_filename = "00japan.txt"  # 你的迷宫文件名
    output_filename = "00japan.world" # 你想生成的world文件名
    # ----------------------------

    # Construct full file paths
    input_file_path = os.path.join(input_dir, input_filename)
    output_file_path = os.path.join(output_dir, output_filename)

    print(f"Reading maze from: {input_file_path}")
    print(f"Output world will be written to: {output_file_path}")

    # Read the maze file
    try:
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
        # sys.exit(1) # 不退出，只打印错误
        return # 遇到错误时直接返回
    except Exception as e:
        print(f"Error reading input file '{input_file_path}': {e}")
        # sys.exit(1) # 不退出，只打印错误
        return # 遇到错误时直接返回


    # Parse the maze data, removing empty lines and stripping whitespace
    maze_data = [list(line.strip()) for line in lines if line.strip()]

    # Validate maze dimensions (expected 33x33)
    expected_size = 33
    if len(maze_data) != expected_size or any(len(row) != expected_size for row in maze_data):
        print(f"Warning: Maze data size is not {expected_size}x{expected_size}. Found {len(maze_data)}x{len(maze_data[0])}. Proceeding anyway...")
        # 如果尺寸不对，可以选择在这里退出，或者继续尝试生成（可能会出错）

    # Generate Gazebo World SDF content
    # You can adjust unit, wall_height, wall_thickness here if needed
    world_sdf_content = generate_world_sdf(maze_data, unit=1.0, wall_height=0.5, wall_thickness=0.1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the generated SDF content to the output file
    try:
        with open(output_file_path, 'w') as f:
            f.write(world_sdf_content)
        print(f"Successfully generated Gazebo world file: {output_file_path}")
    except IOError as e:
        print(f"Error writing to output file '{output_file_path}': {e}")
        # sys.exit(1) # 不退出，只打印错误
    except Exception as e:
        print(f"An unexpected error occurred while writing the output file: {e}")
        # sys.exit(1) # 不退出，只打印错误


if __name__ == "__main__":
    main()
