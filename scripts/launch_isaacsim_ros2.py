
import os
import sys
import argparse
import numpy as np
from isaacsim import SimulationApp

# Initialize SimulationApp
config = {
    "headless": False,
    "width": 1280,
    "height": 720,
    "enable_streaming": False,
    "extensions": [
        "isaacsim.ros2.bridge",
        "isaacsim.robot_motion.motion_generation"
    ]
}
simulation_app = SimulationApp(config)

import omni.graph.core as og
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path

# Define paths
# Assuming running from workspace root
ASSETS_DIR = os.path.abspath("assets")
SCENE_PATH = os.path.join(ASSETS_DIR, "ED305_scene/ED305.usd")
ROBOT_PATH = os.path.join(ASSETS_DIR, "franka_panda/franka_panda_arm.usd")

# Prim paths
ROBOT_ROOT_PRIM = "/World/Franka"
ROBOT_PRIM = f"{ROBOT_ROOT_PRIM}/panda"

class IsaacSimActionGraph:
    """Action Graph for Isaac Sim simulation pipeline with ROS2 integration"""

    def __init__(self, task_name: str, robot_prim_path: str):
        self.task_name = task_name
        self.robot_prim_path = robot_prim_path
        self.graph_handle = None
        self.nodes = {}

    def create_action_graph(self) -> None:
        """Create simulation pipeline action graph with ROS2 integration"""

        # Define node creation configurations
        keys = og.Controller.Keys

        # Define topics to match ROS2Environment defaults
        # ROS2Environment defaults:
        # rgb_topic: '/rgb'
        # joint_states_topic (actually eef pose): '/eef_states'
        # gripper_topic: '/gripper_width'
        # action_topic: '/joint_states'

        (graph_handle, nodes, _, _) = og.Controller.edit(
            {
                "graph_path": "/World/ROS_JointStates",
                "evaluator_name": "execution",
            },
            {
                keys.CREATE_NODES: [
                    # Core timing and context nodes
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                    # ROS2 joint state nodes
                    ("PublisherJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscriberJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),

                    # Robot control
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),

                    # End-effector pose reading
                    ("end_effector_translate", "omni.graph.nodes.ReadPrimAttribute"),
                    ("end_effector_rotate", "omni.graph.nodes.ReadPrimAttribute"),
                    ("break_3_vector", "omni.graph.nodes.BreakVector3"),
                    ("break_4_vector", "omni.graph.nodes.BreakVector4"),

                    # ROS2 pose publisher
                    ("ros2_publisher", "isaacsim.ros2.bridge.ROS2Publisher"),

                    # Gripper width calculation
                    ("read_prim_attribute", "omni.graph.nodes.ReadPrimAttribute"),
                    ("read_prim_attribute_01", "omni.graph.nodes.ReadPrimAttribute"),
                    ("subtract", "omni.graph.nodes.Subtract"),
                    ("magnitude", "omni.graph.nodes.Magnitude"),
                    ("ros2_publisher_01", "isaacsim.ros2.bridge.ROS2Publisher"),

                    # Camera rendering and publishing
                    ("isaac_run_one_simulation_frame", "isaacsim.core.nodes.OmnIsaacRunOneSimulationFrame"),
                    ("isaac_create_render_product", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("ros2_camera_helper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],

                keys.SET_VALUES: [
                    # Context configuration
                    ("Context.inputs:domain_id", 0),
                    ("Context.inputs:useDomainIDEnvVar", True),

                    # Joint state publisher (Actual joint states of the robot)
                    ("PublisherJointState.inputs:topicName", "/robot_joint_states"), # Renamed to avoid conflict with action topic
                    ("PublisherJointState.inputs:nodeNamespace", ""),
                    ("PublisherJointState.inputs:targetPrim", f"{self.robot_prim_path}/root_joint"),

                    # Joint state subscriber (Actions from policy)
                    ("SubscriberJointState.inputs:topicName", "/joint_states"), # Matches ROS2Environment action_topic
                    ("SubscriberJointState.inputs:nodeNamespace", ""),

                    # Articulation controller target
                    ("ArticulationController.inputs:targetPrim", self.robot_prim_path),

                    # End-effector translate reader
                    ("end_effector_translate.inputs:name", "xformOp:translate"),
                    ("end_effector_translate.inputs:prim", f"{self.robot_prim_path}/panda_link7"),
                    ("end_effector_translate.inputs:usePath", False),

                    # End-effector rotation reader
                    ("end_effector_rotate.inputs:name", "xformOp:orient"),
                    ("end_effector_rotate.inputs:prim", f"{self.robot_prim_path}/panda_link7"),
                    ("end_effector_rotate.inputs:usePath", False),

                    # Pose publisher (EEF Pose)
                    ("ros2_publisher.inputs:topicName", "/eef_states"), # Matches ROS2Environment joint_states_topic (which expects Pose)
                    ("ros2_publisher.inputs:messageName", "Pose"),
                    ("ros2_publisher.inputs:messagePackage", "geometry_msgs"),

                    # Gripper width readers
                    ("read_prim_attribute.inputs:name", "xformOp:translate"),
                    ("read_prim_attribute.inputs:prim", f"{self.robot_prim_path}/panda_leftfinger"),
                    ("read_prim_attribute.inputs:usePath", False),

                    ("read_prim_attribute_01.inputs:name", "xformOp:translate"),
                    ("read_prim_attribute_01.inputs:prim", f"{self.robot_prim_path}/panda_rightfinger"),
                    ("read_prim_attribute_01.inputs:usePath", False),

                    # Gripper width publisher
                    ("ros2_publisher_01.inputs:topicName", "/gripper_width"), # Matches ROS2Environment gripper_topic
                    ("ros2_publisher_01.inputs:messageName", "Float64"),
                    ("ros2_publisher_01.inputs:messagePackage", "std_msgs"),

                    # Camera render product
                    ("isaac_create_render_product.inputs:cameraPrim", f"{self.robot_prim_path}/panda_link7/gopro_link/camera"),
                    ("isaac_create_render_product.inputs:width", 224),
                    ("isaac_create_render_product.inputs:height", 224),

                    # Camera helper
                    ("ros2_camera_helper.inputs:topicName", "/rgb"), # Matches ROS2Environment rgb_topic
                    ("ros2_camera_helper.inputs:type", "rgb"),
                    ("ros2_camera_helper.inputs:frameSkipCount", 0),
                ],

                keys.CONNECT: [
                    # Execution flow
                    ("OnPlaybackTick.outputs:tick", "PublisherJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscriberJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "end_effector_translate.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "end_effector_rotate.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "read_prim_attribute.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "read_prim_attribute_01.inputs:usdTimecode"),
                    ("OnPlaybackTick.outputs:tick", "isaac_run_one_simulation_frame.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ros2_publisher.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ros2_publisher_01.inputs:execIn"),

                    # Context propagation
                    ("Context.outputs:context", "PublisherJointState.inputs:context"),
                    ("Context.outputs:context", "SubscriberJointState.inputs:context"),
                    ("Context.outputs:context", "ros2_publisher.inputs:context"),
                    ("Context.outputs:context", "ros2_publisher_01.inputs:context"),
                    ("Context.outputs:context", "ros2_camera_helper.inputs:context"),

                    # Time data
                    ("ReadSimTime.outputs:simulationTime", "PublisherJointState.inputs:timeStamp"),

                    # Joint control flow
                    ("SubscriberJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscriberJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscriberJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscriberJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),

                    # End-effector pose processing
                    ("end_effector_translate.outputs:value", "break_3_vector.inputs:tuple"),
                    ("end_effector_rotate.outputs:value", "break_4_vector.inputs:tuple"),

                    # Position components
                    ("break_3_vector.outputs:x", "ros2_publisher.inputs:position:x"),
                    ("break_3_vector.outputs:y", "ros2_publisher.inputs:position:y"),
                    ("break_3_vector.outputs:z", "ros2_publisher.inputs:position:z"),

                    # Orientation components
                    ("break_4_vector.outputs:x", "ros2_publisher.inputs:orientation:x"),
                    ("break_4_vector.outputs:y", "ros2_publisher.inputs:orientation:y"),
                    ("break_4_vector.outputs:z", "ros2_publisher.inputs:orientation:z"),
                    ("break_4_vector.outputs:w", "ros2_publisher.inputs:orientation:w"),

                    # Gripper width calculation
                    ("read_prim_attribute.outputs:value", "subtract.inputs:a"),
                    ("read_prim_attribute_01.outputs:value", "subtract.inputs:b"),
                    ("subtract.outputs:difference", "magnitude.inputs:input"),
                    ("magnitude.outputs:magnitude", "ros2_publisher_01.inputs:data"),

                    # Camera pipeline
                    ("isaac_run_one_simulation_frame.outputs:step", "isaac_create_render_product.inputs:execIn"),
                    ("isaac_create_render_product.outputs:renderProductPath", "ros2_camera_helper.inputs:renderProductPath"),
                    ("isaac_create_render_product.outputs:execOut", "ros2_camera_helper.inputs:execIn"),
                ],
            },
        )

        self.graph_handle = graph_handle
        self.nodes = nodes
        print(f"Created action graph for task: {self.task_name}")

    def start(self) -> None:
        """Start action graph"""
        if self.graph_handle:
            self.graph_handle.play()
            print(f"Started action graph for {self.task_name}")

def main():
    # Create World
    world = World(stage_units_in_meters=1.0)
    
    # Add Scene
    if os.path.exists(SCENE_PATH):
        add_reference_to_stage(usd_path=SCENE_PATH, prim_path="/World/Scene")
    else:
        print(f"Warning: Scene not found at {SCENE_PATH}")

    # Add Robot
    if os.path.exists(ROBOT_PATH):
        define_prim(ROBOT_ROOT_PRIM, "Xform")
        add_reference_to_stage(usd_path=ROBOT_PATH, prim_path=ROBOT_ROOT_PRIM)
    else:
        print(f"Error: Robot not found at {ROBOT_PATH}")
        return

    # Initialize Action Graph
    action_graph = IsaacSimActionGraph("ros2_bridge", ROBOT_PRIM)
    action_graph.create_action_graph()
    
    # Reset world to apply changes
    world.reset()
    
    # Start Action Graph
    action_graph.start()

    print("Isaac Sim ROS2 Bridge Started. Press Ctrl+C to exit.")
    
    # Simulation Loop
    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
