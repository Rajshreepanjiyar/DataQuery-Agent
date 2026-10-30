
from pathlib import Path
from graph import agent_graph

# Save the image in the main project directory
image_path = Path(__file__).parent / "agent_workflow.png"

# Generate only if the image does not already exist
if image_path.exists():
    print("Workflow diagram already exists. Skipping generation.")
else:
    png_data = agent_graph.get_graph().draw_mermaid_png()
    image_path.write_bytes(png_data)
    print(f"Workflow diagram saved at: {image_path}")
