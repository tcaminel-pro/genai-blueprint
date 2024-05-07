from diagrams import Diagram  # noqa: I001
from diagrams import Diagram, Edge, Node
from diagrams.programming.flowchart import Action, Decision, Inspection

#https://diagrams.mingrammer.com/docs/nodes/programming


graph_attributes = {
    "fontsize": "9",
    "orientation": "portrait",
    "splines":"spline"
}

# fmt: off
with Diagram("Langchain RAG Graph",show=False,graph_attr=graph_attributes): # It is recommended to set "show" to false to prevent the pop out of the diagram in your image viewer
    routing = Decision("Routing") 
    Action("Question") >> routing 
    routing - Edge() >> Inspection("Document Retrieve 1") >> Inspection("Document Grade")
    routing >> Inspection("Web Search")
    routing - Edge(style="dotted") >> Inspection("Other")
    TB_fields  = Node(shape="oval", label="Question")