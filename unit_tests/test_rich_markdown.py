#!/usr/bin/env python3
"""Test Rich markdown rendering"""

from rich.console import Console
from rich.markdown import Markdown

console = Console()

# Sample markdown content similar to what agents produce
markdown_text = """
# AI Regulation Status in Europe

## Overview
The current status of AI regulation in Europe centers around the **EU Artificial Intelligence Act**.

### Key Points:
- 🚀 **Published**: July 12, 2024
- ✅ **In Force**: August 1, 2024  
- 📅 **Fully Applicable**: August 2, 2026

## Risk-Based Approach

The AI Act uses a **risk-based framework** with four levels:

1. **Minimal Risk** - No special requirements
2. **Limited Risk** - Transparency obligations
3. **High Risk** - Strict compliance requirements
4. **Unacceptable Risk** - Banned applications

### High-Risk AI Systems
High-risk systems face requirements for:
- Training data standards
- Validation and testing
- Human oversight
- Documentation

## Key Stakeholders

| Role | Responsibilities |
|------|-----------------|
| **Providers** | Create and supply AI systems |
| **Deployers** | Use AI systems in operations |

## Implementation Timeline

```
August 2024    -> Act enters into force
February 2025  -> Prohibitions apply
August 2026    -> Full application
```

> **Note**: The AI Act complements existing regulations like GDPR

## Support Structures

- **AI Office**: Enforcement for general-purpose AI
- **AI Board**: Coordination across EU states

---

*Sources: EU Commission, AI Act Official Documentation*
"""

# Test rendering
console.print("\n[bold cyan]Testing Rich Markdown Rendering[/bold cyan]\n")
console.print("=" * 60)

md = Markdown(markdown_text)
console.print(md)

console.print("=" * 60)
console.print("\n[green]✓ Test complete![/green]")
