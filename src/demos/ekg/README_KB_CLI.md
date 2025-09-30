# Interactive EKG (Enhanced Knowledge Graph) CLI

A comprehensive Typer-based CLI for managing a single shared Kuzu knowledge graph created from BAML-structured opportunity data.

## Features

- **Add opportunity data** to a shared knowledge base
- **Execute interactive Cypher queries** with Rich formatting
- **Display database information** including schema and BAM class mapping
- **Export HTML visualizations** with clickable links
- **Delete entire database** with safety confirmations
- **Rich console output** with colors, tables, and progress indicators

## Installation

The CLI is ready to use with the existing project setup. All dependencies are included.

## Usage

### Running the CLI

From the project root directory:

```bash
# Basic help
uv run src/demos/ekg/test_graph_cli.py --help

# Command-specific help
uv run src/demos/ekg/test_graph_cli.py add --help
```

### Available Commands

#### 1. Add Opportunity Data (`add`)

Adds opportunity data from the key-value store to the shared EKG database.

```bash
# Add opportunity data to the shared EKG database
uv run src/demos/ekg/test_graph_cli.py add --key cnes-venus-tma
```

**Features:**
- Loads structured data from the key-value store
- Adds data to shared EKG database in `~/kuzu/ekg_database/`
- Creates database if it doesn't exist
- Auto-deduces graph schema from BAM classes
- Provides addition summary with statistics

**Output:**
- Rich console output with progress indicators
- Detailed addition summary table
- Next steps suggestions

#### 2. Query EKG Database (`query`)

Execute Cypher queries on the EKG database with interactive shell or single queries.

```bash
# Interactive query shell
uv run src/demos/ekg/test_graph_cli.py query

# Execute single query
uv run src/demos/ekg/test_graph_cli.py query --query "MATCH (n) RETURN labels(n)[0], count(n)"
```

**Interactive Features:**
- Type `help` for sample queries
- Type `1`, `2`, etc. to run numbered sample queries
- Type `exit`, `quit`, or `q` to leave
- Rich table formatting for results
- Error handling with helpful messages

**Sample Queries:**
1. `MATCH (n) RETURN labels(n)[0] as NodeType, count(n) as Count`
2. `MATCH (o:Opportunity) RETURN o.name, o.status LIMIT 5`
3. `MATCH (c:Customer)-[:HAS_CONTACT]->(p:Person) RETURN c.name, p.name, p.role LIMIT 5`
4. `MATCH (ro:ReviewedOpportunity)-[:HAS_RISK]->(r:RiskAnalysis) RETURN r.risk_description, r.impact_level LIMIT 3`

#### 3. Display Database Information (`info`)

Shows comprehensive information about the EKG database including schema, statistics, and BAM class mapping.

```bash
uv run src/demos/ekg/test_graph_cli.py info
```

**Information Displayed:**
- **Database Information**: Path, type, storage details
- **Schema Overview**: Node and relationship table counts  
- **Node Statistics**: Count of each node type
- **Relationship Statistics**: Count of each relationship type
- **BAM Class Mapping**: Graph nodes → Python classes
- **Relationship Semantics**: Relationship meanings and directions
- **Query Suggestions**: Ready-to-use query examples

#### 4. Export HTML Visualization (`export-html`)

Exports an interactive D3.js graph visualization and displays a clickable link.

```bash
# Export to /tmp (default)
uv run src/demos/ekg/test_graph_cli.py export-html

# Export to custom directory
uv run src/demos/ekg/test_graph_cli.py export-html --output-dir /home/user/graphs

# Export without opening browser
uv run src/demos/ekg/test_graph_cli.py export-html --no-open
```

**Features:**
- Interactive D3.js visualization
- Zoomable and draggable interface
- Hover tooltips for nodes and edges
- Automatically opens in default browser
- Displays clickable file:// link in terminal

#### 5. Delete EKG Database (`delete`)

Safely deletes the entire shared EKG database with confirmations.

```bash
uv run src/demos/ekg/test_graph_cli.py delete
```

**Safety Features:**
- Shows database statistics before deletion
- Double confirmation prompts
- Deletes ALL opportunity data
- Cannot be undone (intentional safety measure)

## Graph Schema

The CLI uses an auto-deduced graph schema based on BAM classes:

### Node Types

| Graph Node | BAM Class | Description |
|------------|-----------|-------------|
| `ReviewedOpportunity` | `ReviewedOpportunity` | Root node containing the complete reviewed opportunity |
| `Opportunity` | `Opportunity` | Core opportunity information with financial metrics embedded |
| `Customer` | `Customer` | Customer organization details |
| `Person` | `Person` | Individual contacts and team members |
| `Partner` | `Partner` | Partner organization information |
| `RiskAnalysis` | `RiskAnalysis` | Risk assessment and mitigation details |
| `TechnicalApproach` | `TechnicalApproach` | Technical implementation approach and stack |
| `CompetitiveLandscape` | `CompetitiveLandscape` | Competitive positioning and analysis |

### Relationships

| Relationship | Direction | Meaning |
|--------------|-----------|---------|
| `REVIEWS` | ReviewedOpportunity → Opportunity | Review relationship to core opportunity |
| `HAS_CUSTOMER` | Opportunity → Customer | Opportunity belongs to customer |
| `HAS_CONTACT` | Customer → Person | Customer contact persons |
| `HAS_TEAM_MEMBER` | ReviewedOpportunity → Person | Internal team members |
| `HAS_PARTNER` | ReviewedOpportunity → Partner | Partner organizations involved |
| `HAS_RISK` | ReviewedOpportunity → RiskAnalysis | Identified risks and mitigations |
| `HAS_TECH_STACK` | ReviewedOpportunity → TechnicalApproach | Technical implementation approach |
| `HAS_COMPETITION` | ReviewedOpportunity → CompetitiveLandscape | Competitive analysis |

## Data Flow

1. **Structured Data Extraction**: Use BAML extraction commands to create opportunity data in the key-value store
2. **Knowledge Base Creation**: CLI loads data and creates Kuzu graph database
3. **Interactive Querying**: Use Cypher queries to explore the graph
4. **Visualization Export**: Generate interactive HTML visualizations

## File Locations

- **EKG Database**: `~/kuzu/ekg_database/`
- **HTML Exports**: `/tmp/ekg_graph_visualization.html` (default)
- **Key-Value Store**: Managed by the existing KV store infrastructure

## Error Handling

The CLI provides user-friendly error messages for common scenarios:

- Missing opportunity data in key-value store
- Non-existent knowledge bases
- Invalid Cypher queries
- File permission issues
- Database connection errors

## Integration with Existing Workflow

The CLI integrates seamlessly with the existing project structure:

1. **Extract Data**: Use existing BAML extraction commands to populate the key-value store
2. **Add Data**: Use `add` to add opportunity data to the shared EKG database
3. **Query & Analyze**: Use `query` and `info` commands for analysis
4. **Visualize**: Use `export-html` for stakeholder presentations

## Dependencies

The CLI uses the project's existing dependencies:

- **Typer**: CLI framework with rich help
- **Rich**: Console formatting and tables
- **Kuzu**: Graph database
- **Pandas**: Data manipulation for query results
- **BAML Client**: Access to structured data types

## Advanced Usage

### Custom Cypher Queries

The knowledge base supports standard Cypher syntax:

```cypher
-- Find high-impact risks
MATCH (ro:ReviewedOpportunity)-[:HAS_RISK]->(r:RiskAnalysis)
WHERE r.impact_level = 'High'
RETURN ro.start_date, r.risk_description, r.mitigation_strategy

-- Analyze team composition
MATCH (ro:ReviewedOpportunity)-[:HAS_TEAM_MEMBER]->(p:Person)
RETURN p.role, count(p) as team_size
ORDER BY team_size DESC

-- Find opportunities by customer
MATCH (o:Opportunity)-[:HAS_CUSTOMER]->(c:Customer)
RETURN c.name as customer, collect(o.name) as opportunities
```

### Batch Operations

You can script the CLI for batch operations:

```bash
#!/bin/bash
OPPORTUNITIES=("cnes-venus-tma" "project-alpha" "beta-initiative")

# Add all opportunities to the shared EKG database
for opp in "${OPPORTUNITIES[@]}"; do
  echo "Adding opportunity $opp to EKG database..."
  uv run src/demos/ekg/test_graph_cli.py add --key "$opp"
done

# Export single visualization containing all data
echo "Exporting combined EKG visualization..."
uv run src/demos/ekg/test_graph_cli.py export-html --output-dir "./exports" --no-open
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Run from project root directory
2. **Missing Data**: Ensure opportunity data exists in key-value store first
3. **Permission Errors**: Check write permissions for `~/kuzu/ekg_database/` directory
4. **Browser Issues**: Use the displayed file:// link to open manually

### Debug Mode

For detailed debugging, you can modify the CLI to add more verbose logging or run individual components from the original test files.

## Examples in Context

### Complete Workflow Example

```bash
# 1. First, extract structured data (from existing BAML commands)
uv run cli structured-extract-baml "opportunity_reviews/*.md" --force

# 2. Add opportunity data to EKG database
uv run src/demos/ekg/test_graph_cli.py add --key cnes-venus-tma

# 3. Add more opportunities to the same database
uv run src/demos/ekg/test_graph_cli.py add --key project-alpha
uv run src/demos/ekg/test_graph_cli.py add --key beta-initiative

# 4. Explore the combined data
uv run src/demos/ekg/test_graph_cli.py info

# 5. Run interactive queries on all data
uv run src/demos/ekg/test_graph_cli.py query

# 6. Export combined visualization for presentation
uv run src/demos/ekg/test_graph_cli.py export-html --output-dir ./presentations
```

This creates a complete pipeline from document extraction to interactive graph exploration and visualization.