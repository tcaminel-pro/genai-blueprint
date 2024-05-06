from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

with Diagram("Web Service",show=False): # It is recommended to set "show" to false to prevent the pop out of the diagram in your image viewer
    ELB("lb") >> EC2("Production") >> RDS("Accounts")
    ELB("lb") >> EC2("UAT") >> RDS("Accounts")
    ELB("lb") >> EC2("DevDevDevDev") >> RDS("Accounts")