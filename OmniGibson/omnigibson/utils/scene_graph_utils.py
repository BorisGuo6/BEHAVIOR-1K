"""
Scene Graph Representation Utils
"""
import os
import json
from PIL import Image

import networkx as nx

from copy import deepcopy
from math import ceil
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any, Set, Union
from dataclasses import field, dataclass
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.object_states import ContactBodies

def convert_to_serializable(obj):
    '''
    Recursively convert tensors and numpy arrays to lists
    '''
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def get_symbolic_scene_graph(nx_graph: nx.Graph, obj_visibility_dict: Dict[Any, Any]=None) -> Dict[str, List[Dict]]:
    '''
    Get the symbolic scene graph from the nx graph
    '''
    # Convert the nx graph to a serializable format
    symbolic_graph = {
        'nodes': [],
        'edges': []
    }
    
    for node in nx_graph.nodes():
        node_name = node.name
        node_category = node.category if hasattr(node, 'category') else 'System'
        node_data = nx_graph.nodes[node]
        node_parent = node_data['parent'] if 'parent' in node_data else None
        states = convert_to_serializable(node_data['states'])
        symbolic_graph['nodes'].append({
            'name': node_name,
            'category': node_category,
            'states': set(states.keys())
        })
        if obj_visibility_dict is not None and node_name in obj_visibility_dict:
            symbolic_graph['nodes'][-1]['visibility'] = obj_visibility_dict[node_name]
        else:
            symbolic_graph['nodes'][-1]['visibility'] = {}
        if node_parent is not None:
            symbolic_graph['nodes'][-1]['parent'] = node_parent
    
    for u, v, data in nx_graph.edges(data=True):
        edge_states = convert_to_serializable(data.get('states', []))
        symbolic_graph['edges'].append({
            "from": u.name,
            "to": v.name,
            "states": set([state[0] for state in edge_states if state[1]])
        })
    
    return symbolic_graph



def generate_scene_graph_diff(
    prev_graph: Dict[str, List[Dict]],
    new_graph: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    '''
    Generate the diff between two scene graphs, return as the state representation

    Args:
        prev_graph (Dict[str, List[Dict]]): The previous scene graph
        new_graph (Dict[str, List[Dict]]): The new scene graph
    
    Returns:
        Dict[str, List[Dict]]: The diff between the two scene graphs
    '''
    diff_graph = {
        "type": "diff",
        "add": {'nodes': [], 'edges': []},
        "remove": {'nodes': [], 'edges': []},
        "update": {'nodes': [], 'edges': []}
    }

    # node structure: {name: str, states: set[str]}
    # edge structure: {from: str, to: str, states: set[str]}

    # Pre-convert states to sets and create efficient lookups
    prev_nodes = {}
    new_nodes = {}
    
    for node in prev_graph['nodes']:
        prev_nodes[node['name']] = deepcopy(node)
    
    for node in new_graph['nodes']:
        new_nodes[node['name']] = deepcopy(node)
    
    # Get node name sets for efficient set operations
    prev_node_names = set(prev_nodes.keys())
    new_node_names = set(new_nodes.keys())
    
    # Process all node changes in one pass
    added_node_names = new_node_names - prev_node_names
    removed_node_names = prev_node_names - new_node_names
    common_node_names = prev_node_names & new_node_names
    
    # Add new nodes
    diff_graph['add']['nodes'].extend(new_nodes[name] for name in added_node_names)
    
    # Add removed nodes
    diff_graph['remove']['nodes'].extend(prev_nodes[name] for name in removed_node_names)
    
    # Check for updated nodes
    for node_name in common_node_names:
        prev_states = prev_nodes[node_name]['states']
        new_states = new_nodes[node_name]['states']
        # use **other_args to save other args (should remain the same)
        other_args = {k: v for k, v in prev_nodes[node_name].items() if k not in ['states', 'name']}
        
        if prev_states != new_states:
            diff_graph['update']['nodes'].append({
                'name': node_name,
                'states': new_states,
                **other_args
            })
    
    # Pre-convert edge states to sets and create efficient lookups
    prev_edges = {}
    new_edges = {}
    
    for edge in prev_graph['edges']:
        key = (edge['from'], edge['to'])
        prev_edges[key] = deepcopy(edge)
    
    for edge in new_graph['edges']:
        key = (edge['from'], edge['to'])
        new_edges[key] = deepcopy(edge)
    
    # Get edge key sets for efficient set operations
    prev_edge_keys = set(prev_edges.keys())
    new_edge_keys = set(new_edges.keys())
    
    # Process all edge changes in one pass
    added_edge_keys = new_edge_keys - prev_edge_keys
    removed_edge_keys = prev_edge_keys - new_edge_keys
    common_edge_keys = prev_edge_keys & new_edge_keys
    
    # Add new edges
    diff_graph['add']['edges'].extend(new_edges[key] for key in added_edge_keys)
    
    # Add removed edges  
    diff_graph['remove']['edges'].extend(prev_edges[key] for key in removed_edge_keys)
    
    # Check for updated edges
    for edge_key in common_edge_keys:
        prev_states = prev_edges[edge_key]['states']
        new_states = new_edges[edge_key]['states']
        
        if prev_states != new_states:
            edge = new_edges[edge_key]
            diff_graph['update']['edges'].append({
                'from': edge['from'],
                'to': edge['to'],
                'states': new_states
            })
    
    # Check if the diff is empty (no changes)
    if (not diff_graph['add']['nodes'] and not diff_graph['add']['edges'] and
        not diff_graph['remove']['nodes'] and not diff_graph['remove']['edges'] and
        not diff_graph['update']['nodes'] and not diff_graph['update']['edges']):
        diff_graph = {"type": "empty"}
    
    return diff_graph


def generate_state_centric_diff(
    prev_graph: Dict[str, List[Dict]],
    new_graph: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    '''
    Generate a state-centric diff between two scene graphs.
    
    This function implements the new state-centric diff format that tracks
    individual state additions and removals rather than node/edge updates.
    
    Args:
        prev_graph: The previous scene graph with 'nodes' and 'edges' keys
        new_graph: The new scene graph with 'nodes' and 'edges' keys
    
    Returns:
        Dict with 'add' and 'remove' keys containing state-level changes
    '''
    diff = {
        "add": {'nodes': [], 'edges': []},
        "remove": {'nodes': [], 'edges': []}
    }
    
    # Convert node lists to dictionaries for efficient lookup
    prev_nodes = {node['name']: set(node.get('states', [])) for node in prev_graph['nodes']}
    new_nodes = {node['name']: set(node.get('states', [])) for node in new_graph['nodes']}

    prev_nodes_category = {node['name']: node.get('category', None) for node in prev_graph['nodes']}
    new_nodes_category = {node['name']: node.get('category', None) for node in new_graph['nodes']}

    prev_nodes_parent = {node['name']: node.get('parent', None) for node in prev_graph['nodes']}
    new_nodes_parent = {node['name']: node.get('parent', None) for node in new_graph['nodes']}
    
    # Process node state changes
    all_node_names = set(prev_nodes.keys()) & set(new_nodes.keys())

    removed_nodes_names = set(prev_nodes.keys()) - set(new_nodes.keys())
    added_nodes_names = set(new_nodes.keys()) - set(prev_nodes.keys())

    for node_name in removed_nodes_names:
        diff['remove']['nodes'].append({
            'name': node_name,
            'states': [],
            'category': prev_nodes_category[node_name],
            'parent': prev_nodes_parent[node_name]
        })
    
    for node_name in added_nodes_names:
        node_parent = new_nodes_parent[node_name]
        node_category = new_nodes_category[node_name]
        # assert node_parent is not None, f"Added node {node_name} has no parent"
        if node_parent is None:
            node_parent = []

        elif node_category == 'System':
            # 1. If cooked, we only care about the system 
            if 'cooked' in node_name.lower():
                idealized_parent = node_name.replace('cooked__', '')
                node_parent = [idealized_parent]
            else:
                # 2. We search if there if half objects exist
                complete_string = f"{node_parent[0]}"
                half_string = f"half_{node_parent[0]}"
                updated_node_parent = [parent for parent in removed_nodes_names if half_string in parent]
                if len(updated_node_parent) == 0:
                    updated_node_parent = [parent for parent in removed_nodes_names if complete_string in parent]
                assert len(updated_node_parent) > 0, f"Added node {node_name} has no corresponding parent"
                node_parent = updated_node_parent

        diff['add']['nodes'].append({
            'name': node_name,
            'states': [],
            'category': node_category,
            'parent': node_parent
        })
    
    for node_name in all_node_names:
        prev_states = prev_nodes.get(node_name, set())
        new_states = new_nodes.get(node_name, set())
        node_category = prev_nodes_category.get(node_name, None)
        
        # States that were added (present in new but not in prev)
        added_states = new_states - prev_states
        if added_states:
            diff['add']['nodes'].append({
                'name': node_name,
                'states': list(added_states),
                'category': node_category,
                'parent': new_nodes_parent[node_name]
            })
        
        # States that were removed (present in prev but not in new)
        removed_states = prev_states - new_states
        if removed_states:
            diff['remove']['nodes'].append({
                'name': node_name,
                'states': list(removed_states),
                'category': node_category,
                'parent': prev_nodes_parent[node_name]
            })
    
    
    # Convert edge lists to dictionaries for efficient lookup
    prev_edges = {}
    new_edges = {}
    
    for edge in prev_graph['edges']:
        key = (edge['from'], edge['to'])
        prev_edges[key] = set(edge.get('states', []))
    
    for edge in new_graph['edges']:
        key = (edge['from'], edge['to'])
        new_edges[key] = set(edge.get('states', []))
    
    # Process edge state changes
    all_edge_keys = set(prev_edges.keys()) | set(new_edges.keys())
    
    for edge_key in all_edge_keys:
        prev_states = prev_edges.get(edge_key, set())
        new_states = new_edges.get(edge_key, set())
        
        # States that were added
        added_states = new_states - prev_states
        if added_states:
            diff['add']['edges'].append({
                'from': edge_key[0],
                'to': edge_key[1],
                'states': list(added_states)
            })
        
        # States that were removed
        removed_states = prev_states - new_states
        if removed_states:
            diff['remove']['edges'].append({
                'from': edge_key[0],
                'to': edge_key[1],
                'states': list(removed_states)
            })
    
    # Check if the diff is empty (no state changes)
    if (not diff['add']['nodes'] and not diff['add']['edges'] and
        not diff['remove']['nodes'] and not diff['remove']['edges']):
        return {"type": "empty"}
    
    return diff

class SceneGraphWriter:
    output_path: str
    interval: int
    buffer_size: int
    buffer: List[Dict]

    def __init__(self, output_path: str, interval: int=1000, buffer_size: int=1000, write_full_graph_only: bool=False):
        self.output_path = output_path
        self.interval = interval
        self.batch_step = 0
        self.buffer_size = buffer_size
        self.buffer = []
        self.prev_graph = None
        self.current_graph = None
        self.prev_time = -1
        self.write_full_graph_only = write_full_graph_only

    def step(self, graph: nx.Graph, frame_step: int, obj_visibility_dict: Dict[str, Dict[str, List[int]]]=None):
        '''
        Step the scene graph writer
        '''
        symbolic_graph = get_symbolic_scene_graph(graph, obj_visibility_dict)
        self.current_graph = symbolic_graph

        self.batch_step += 1

        # if this is the first graph, just write the full graph
        if self.write_full_graph_only or \
          (self.prev_graph is None or self.batch_step == self.interval or self.prev_time == 0):
            data = deepcopy(symbolic_graph)
            data['type'] = 'full'
            if self.batch_step == self.interval:
                self.batch_step = 0
        # otherwise, write the diff
        else:
            data = generate_scene_graph_diff(self.prev_graph, self.current_graph)

        complete_data = {str(frame_step): data}
        self.buffer.append(complete_data)
        self.prev_graph = deepcopy(symbolic_graph)
        self.current_graph = None

        if len(self.buffer) >= self.buffer_size:
            self._flush()

        self.prev_time = frame_step
            
    def _flush(self):
        '''
        Flush the buffer to the output file
        '''
        if not os.path.exists(self.output_path):
            data = {}
        else:
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        assert type(data) == dict, "Old data must be a dictionary"
        
        for buffer_item in self.buffer:
            serializable_item = convert_to_serializable(buffer_item)
            data.update(serializable_item)
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f)
        self.buffer = []

    def close(self):
        '''
        Close the scene graph writer
        '''
        self._flush()


class FrameWriter:
    """
    A utility class for writing RGB frames to PNG files with zero-padded filenames.
    Similar to video writers but saves individual PNG frames instead.
    Uses buffering to minimize I/O operations for better performance.
    """
    
    def __init__(self, output_dir, filename_prefix="", buffer_size=1000):
        """
        Initialize the frame writer.
        
        Args:
            output_dir (str): Directory where PNG frames will be saved
            filename_prefix (str): Optional prefix for filenames (default: "")
            buffer_size (int): Number of frames to buffer before auto-flushing (default: 100)
        """
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.buffer_size = buffer_size
        
        # Buffer to store frames before writing
        self.frame_buffer = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def append_data(self, frame_data, frame_count):
        """
        Buffer a frame for later writing.
        
        Args:
            frame_data (numpy.ndarray): RGB frame data to save
            frame_count (int): Frame number to save
        """
        # Convert to uint8 if needed
        if frame_data.dtype != 'uint8':
            frame_data = frame_data.astype('uint8')
        
        # Add frame to buffer
        self.frame_buffer.append((frame_data.copy(), frame_count))
        
        # Auto-flush if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """
        Write all buffered frames to disk.
        """
        if not self.frame_buffer:
            return
            
        # Write all buffered frames
        for frame_data, frame_count in self.frame_buffer:
            # Generate filename with zero-padded frame number
            filename = f"{self.filename_prefix}{frame_count:05d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the frame
            Image.fromarray(frame_data).save(filepath)
        
        # Clear the buffer
        self.frame_buffer.clear()
    
    def close(self):
        """
        Flush any remaining buffered frames and close the frame writer.
        """
        self.flush()

@dataclass
class CustomizedUnaryStates:
    ...

@dataclass
class CustomizedBinaryStates:
    LeftGrasping: Callable[[Any, Any], bool] = field(init=False, repr=False)
    RightGrasping: Callable[[Any, Any], bool] = field(init=False, repr=False)
    # LeftContact: Callable[[Any, Any], bool] = field(init=False, repr=False)
    # RightContact: Callable[[Any, Any], bool] = field(init=False, repr=False)


    def __post_init__(self):
        self.LeftGrasping = lambda obj, candidate_obj=None: (
            obj.is_grasping(arm="left", candidate_obj=candidate_obj).value == 1
            if hasattr(obj, "is_grasping") and "left" in getattr(obj, "arm_names", ())
            else False
        )
        self.RightGrasping = lambda obj, candidate_obj=None: (
            obj.is_grasping(arm="right", candidate_obj=candidate_obj).value == 1
            if hasattr(obj, "is_grasping") and "right" in getattr(obj, "arm_names", ())
            else False
        )
        # self.LeftContact = lambda obj, candidate_obj=None: (
        #     len(candidate_obj.states[ContactBodies].get_value().intersection(obj.finger_links["left"])) > 0
        #     if isinstance(obj, ManipulationRobot) and "left" in getattr(obj, "arm_names", ())
        #     else False
        # )
        # self.RightContact = lambda obj, candidate_obj=None: (
        #     len(candidate_obj.states[ContactBodies].get_value().intersection(obj.finger_links["right"])) > 0
        #     if isinstance(obj, ManipulationRobot) and "right" in getattr(obj, "arm_names", ())
        #     else False
        # )
