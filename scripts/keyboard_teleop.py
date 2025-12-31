"""
Keyboard teleoperation for Isaac Sim Franka arm.
Supports both joint-space and Cartesian (end-effector) control.
"""

import carb
import numpy as np
from isaacsim.core.utils.types import ArticulationAction


class KeyboardTeleop:
    """Keyboard teleoperator for Franka arm.
    
    Controls:
    =========
    End-Effector Position (Cartesian):
      - Arrow Keys (↑↓←→): Move in X-Y plane
      - Page Up/Dn: Move in Z (up/down)
      - Ctrl + Arrow: Rotate around Z (yaw)
      - Shift + Arrow: Rotate around X/Y (pitch/roll)
    
    Joint Space:
      - [1-7]: Select joint (1=base, 7=gripper)
      - +/-: Increase/decrease selected joint angle
      - Space: Reset to neutral pose
      - G: Toggle gripper open/close
    
    Modes:
      - Press 'C': Cartesian (end-effector) mode
      - Press 'J': Joint space mode
      - Press 'H': Print help
      - Press 'ESC': Exit
    """
    
    def __init__(self, panda, lula_solver, art_kine_solver, world=None):
        """Initialize keyboard teleop controller.
        
        Args:
            panda: SingleManipulator instance
            lula_solver: LulaKinematicsSolver instance
            art_kine_solver: ArticulationKinematicsSolver instance
            world: Isaac Sim World instance (for stepping)
        """
        self.panda = panda
        self.lula_solver = lula_solver
        self.art_kine_solver = art_kine_solver
        self.world = world
        
        self.mode = "cartesian"  # "cartesian" or "joint"
        self.selected_joint = 0
        self.gripper_closed = False
        
        # Cartesian increments
        self.xy_step = 0.01  # meters
        self.z_step = 0.01   # meters
        self.rot_step = 0.05  # radians
        
        # Joint increments
        self.joint_step = 0.05  # radians
        
        # Neutral pose (7 DOF)
        self.neutral_pose = np.array([0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, 0.0])
        
        # Input interface for keyboard
        self.input = carb.input.acquire_input_interface()
        self._print_help()
    
    def _print_help(self):
        """Print teleop controls."""
        print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    KEYBOARD TELEOPERATION CONTROLS                     ║
╠════════════════════════════════════════════════════════════════════════╣
║ MODE SELECTION:                                                        ║
║   C = Cartesian (End-Effector)  |  J = Joint Space  |  H = Help       ║
║                                                                        ║
║ CARTESIAN MODE (End-Effector Control):                                ║
║   ↑↓←→      = XY movement (±{:.3f}m)                                   ║
║   PgUp/Dn  = Z movement (±{:.3f}m)                                    ║
║   Ctrl+↑↓← = Yaw rotation (±{:.3f}rad)                               ║
║   Shift+↑↓← = Pitch/Roll rotation (±{:.3f}rad)                       ║
║                                                                        ║
║ JOINT SPACE MODE (Individual Joint Control):                          ║
║   1-7       = Select joint (1=shoulder, 7=wrist)                      ║
║   +/-       = Increment/Decrement selected joint (±{:.3f}rad)         ║
║   Space     = Reset to neutral                                        ║
║   G         = Toggle gripper open/close                               ║
║                                                                        ║
║ GLOBAL:                                                                ║
║   ESC       = Exit teleoperation                                      ║
╚════════════════════════════════════════════════════════════════════════╝
        """.format(
            self.xy_step, self.z_step, self.rot_step, self.rot_step, self.joint_step
        ))
    
    def step(self):
        """Process keyboard input and update robot state.
        
        Returns:
            bool: False if ESC pressed (exit signal), True otherwise
        """
        if not self.input:
            return True
        
        # Check for mode switches
        if self.input.is_key_held("C"):
            self.mode = "cartesian"
            print("[TeleOp] Switched to CARTESIAN mode")
        elif self.input.is_key_held("J"):
            self.mode = "joint"
            print("[TeleOp] Switched to JOINT SPACE mode")
        elif self.input.is_key_held("H"):
            self._print_help()
        elif self.input.is_key_held("ESCAPE"):
            return False
        
        if self.mode == "cartesian":
            self._handle_cartesian_input()
        else:
            self._handle_joint_input()
        
        return True
    
    def _handle_cartesian_input(self):
        """Handle cartesian (end-effector) position/orientation control."""
        if not self.input:
            return
        
        # Get current EE pose
        robot_pos, robot_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(robot_position=robot_pos, robot_orientation=robot_quat)
        ee_pos, ee_T = self.art_kine_solver.compute_end_effector_pose()
        
        from scipy.spatial.transform import Rotation as R
        ee_rot = R.from_matrix(ee_T[:3, :3])
        ee_quat_xyzw = ee_rot.as_quat()
        ee_quat_wxyz = np.array([ee_quat_xyzw[3], ee_quat_xyzw[0], ee_quat_xyzw[1], ee_quat_xyzw[2]])
        
        delta_pos = np.zeros(3)
        delta_rot = np.zeros(3)  # Euler angles
        
        # XY movement
        if self.input.is_key_held("UP"):
            delta_pos[0] += self.xy_step
        if self.input.is_key_held("DOWN"):
            delta_pos[0] -= self.xy_step
        if self.input.is_key_held("RIGHT"):
            delta_pos[1] += self.xy_step
        if self.input.is_key_held("LEFT"):
            delta_pos[1] -= self.xy_step
        
        # Z movement
        if self.input.is_key_held("PAGE_UP"):
            delta_pos[2] += self.z_step
        if self.input.is_key_held("PAGE_DOWN"):
            delta_pos[2] -= self.z_step
        
        # Rotation (yaw)
        if self.input.is_key_held("LEFT_CONTROL"):
            if self.input.is_key_held("RIGHT"):
                delta_rot[2] -= self.rot_step  # Yaw CCW
            if self.input.is_key_held("LEFT"):
                delta_rot[2] += self.rot_step  # Yaw CW
        
        # Rotation (pitch/roll)
        if self.input.is_key_held("LEFT_SHIFT"):
            if self.input.is_key_held("UP"):
                delta_rot[1] += self.rot_step  # Pitch up
            if self.input.is_key_held("DOWN"):
                delta_rot[1] -= self.rot_step  # Pitch down
        
        # Apply deltas
        target_pos = ee_pos + delta_pos
        
        if np.any(delta_rot != 0):
            current_euler = ee_rot.as_euler('xyz')
            target_euler = current_euler + delta_rot
            new_rot = R.from_euler('xyz', target_euler)
            new_quat_xyzw = new_rot.as_quat()
            target_quat_wxyz = np.array([new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]])
        else:
            target_quat_wxyz = ee_quat_wxyz
        
        # Apply IK
        self._apply_ik(target_pos, target_quat_wxyz)
    
    def _handle_joint_input(self):
        """Handle joint space control."""
        if not self.input:
            return
        
        current_pose = self.panda.get_joint_positions()
        target_pose = current_pose.copy()
        
        # Joint selection
        for i in range(1, 8):
            if self.input.is_key_held(str(i)):
                self.selected_joint = i - 1
                print(f"[TeleOp] Selected joint {i}")
                break
        
        # Increment/decrement selected joint
        if self.input.is_key_held("PLUS") or self.input.is_key_held("EQUAL"):
            target_pose[self.selected_joint] += self.joint_step
            print(f"[TeleOp] Joint {self.selected_joint + 1}: {target_pose[self.selected_joint]:.3f}")
        if self.input.is_key_held("MINUS"):
            target_pose[self.selected_joint] -= self.joint_step
            print(f"[TeleOp] Joint {self.selected_joint + 1}: {target_pose[self.selected_joint]:.3f}")
        
        # Reset to neutral
        if self.input.is_key_held("SPACE"):
            target_pose = self.neutral_pose.copy()
            print("[TeleOp] Reset to neutral pose")
        
        # Gripper toggle
        if self.input.is_key_held("G"):
            self.gripper_closed = not self.gripper_closed
            if self.gripper_closed:
                self.panda.gripper.close()
                print("[TeleOp] Gripper CLOSED")
            else:
                self.panda.gripper.open()
                print("[TeleOp] Gripper OPEN")
        
        # Apply joint positions
        self.panda.set_joint_positions(target_pose, np.arange(len(target_pose)))
    
    def _apply_ik(self, target_pos, target_quat_wxyz):
        """Apply IK to reach target pose."""
        robot_pos, robot_quat = self.panda.get_world_pose()
        self.lula_solver.set_robot_base_pose(robot_position=robot_pos, robot_orientation=robot_quat)
        
        action, success = self.art_kine_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_quat_wxyz
        )
        
        if success:
            self.panda.set_joint_positions(action.joint_positions, np.arange(7))
        else:
            print("[TeleOp] WARNING: IK failed for target pose")
