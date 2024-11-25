# From https://github.com/nv-tlabs/LLaMA-Mesh/
# For use with https://huggingface.co/c01zaut/LLaMA-Mesh-rk3588-1.1.1
from trimesh.exchange.gltf import export_glb
import trimesh
import numpy as np
import tempfile

def apply_gradient_color(mesh_text):
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
    Args:
        mesh_text (str): The input mesh in OBJ format as a string.
    Returns:
        str: Path to the GLB file with gradient colors applied.
    """
    # Load the mesh
    temp_file =  tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
    with open(temp_file+".obj", "w") as f:
        f.write(mesh_text)
    # return temp_file
    mesh = trimesh.load_mesh(temp_file+".obj", file_type='obj')

    # Get vertex coordinates
    vertices = mesh.vertices
    y_values = vertices[:, 1]  # Y-axis values

    # Normalize Y values to range [0, 1] for color mapping
    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    # Generate colors: Map normalized Y values to RGB gradient (e.g., blue to red)
    colors = np.zeros((len(vertices), 4))  # RGBA
    colors[:, 0] = y_normalized  # Red channel
    colors[:, 2] = 1 - y_normalized  # Blue channel
    colors[:, 3] = 1.0  # Alpha channel (fully opaque)

    # Attach colors to mesh vertices
    mesh.visual.vertex_colors = colors

    # Export to GLB format
    glb_path = temp_file+".glb"
    with open(glb_path, "wb") as f:
        f.write(export_glb(mesh))
    
    return glb_path

def visualize_mesh(mesh_text):
    """
    Convert the provided 3D mesh text into a visualizable format.
    This function assumes the input is in OBJ format.
    """
    temp_file = "temp_mesh.obj"
    with open(temp_file, "w") as f:
        f.write(mesh_text)
    return temp_file