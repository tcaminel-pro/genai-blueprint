"""Utility functions for displaying available CLI configurations.

This module provides functions to display available configurations for different
agent types in a user-friendly format when invalid configurations are specified.
"""

import yaml
from pathlib import Path
from typing import Any


def display_react_agent_configs() -> None:
    """Display available React Agent configurations in a formatted way."""
    config_file = Path("config/demos/react_agent.yaml")
    
    print("📋 Available React Agent Configurations:")
    print("=" * 50)
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'react_agent_demos' in config:
            for i, demo in enumerate(config['react_agent_demos'], 1):
                name = demo.get('name', f'Demo {i}')
                print(f'\n🎯 {i}. "{name}"')
                
                # Show tools if available
                if 'tools' in demo:
                    print(f'   📦 Tools: {len(demo["tools"])} configured')
                    for tool in demo['tools'][:2]:  # Show first 2 tools
                        if isinstance(tool, dict):
                            if 'function' in tool:
                                func_name = tool['function'].split(':')[-1]
                                print(f'      • {func_name} (function)')
                            elif 'factory' in tool:
                                factory_name = tool['factory'].split(':')[-1]
                                print(f'      • {factory_name} (factory)')
                        else:
                            print(f'      • {tool}')
                    if len(demo['tools']) > 2:
                        print(f'      • ... and {len(demo["tools"]) - 2} more')
                
                # Show MCP servers if available
                if 'mcp_servers' in demo:
                    servers = ', '.join(demo['mcp_servers'])
                    print(f'   🔗 MCP Servers: {servers}')
                
                # Show examples if available
                if 'examples' in demo:
                    print(f'   💡 Example prompts ({len(demo["examples"])} available):')
                    for example in demo['examples'][:2]:  # Show first 2 examples
                        print(f'      • "{example}"')
                    if len(demo['examples']) > 2:
                        print(f'      • ... and {len(demo["examples"]) - 2} more')
        
        print('\n' + '=' * 50)
        print('💡 Usage: uv run cli react-agent --config "<configuration_name>"')
        print('   Example: uv run cli react-agent --config "Weather"')
        
    except FileNotFoundError:
        print(f'❌ Configuration file not found: {config_file}')
    except yaml.YAMLError as e:
        print(f'❌ Error parsing YAML file: {e}')
    except Exception as e:
        print(f'❌ Error: {e}')


def display_smolagents_configs() -> None:
    """Display available SmolAgents configurations in a formatted way."""
    config_file = Path("config/demos/codeact_agent.yaml")
    
    print("📋 Available SmolAgents Configurations:")
    print("=" * 50)
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'codeact_agent_demos' in config:
            for i, demo in enumerate(config['codeact_agent_demos'], 1):
                name = demo.get('name', f'Demo {i}')
                print(f'\n🎯 {i}. "{name}"')
                
                # Show tools if available
                if 'tools' in demo:
                    print(f'   📦 Tools: {len(demo["tools"])} configured')
                    for tool in demo['tools'][:2]:  # Show first 2 tools
                        if isinstance(tool, dict):
                            if 'class' in tool:
                                class_name = tool['class'].split(':')[-1] if ':' in tool['class'] else tool['class']
                                print(f'      • {class_name}')
                            elif 'function' in tool:
                                func_name = tool['function'].split(':')[-1]
                                print(f'      • {func_name} (function)')
                            elif 'factory' in tool:
                                factory_name = tool['factory'].split(':')[-1]
                                print(f'      • {factory_name} (factory)')
                        else:
                            print(f'      • {tool}')
                    if len(demo['tools']) > 2:
                        print(f'      • ... and {len(demo["tools"]) - 2} more')
                
                # Show MCP servers if available
                if 'mcp_servers' in demo:
                    servers = ', '.join(demo['mcp_servers'])
                    print(f'   🔗 MCP Servers: {servers}')
                
                # Show authorized imports if available
                if 'authorized_imports' in demo:
                    imports = ', '.join(demo['authorized_imports'][:3])  # Show first 3
                    if len(demo['authorized_imports']) > 3:
                        imports += f", ... and {len(demo['authorized_imports']) - 3} more"
                    print(f'   📚 Authorized Imports: {imports}')
                
                # Show examples if available
                if 'examples' in demo:
                    print(f'   💡 Example prompts ({len(demo["examples"])} available):')
                    for example in demo['examples'][:2]:  # Show first 2 examples
                        print(f'      • "{example}"')
                    if len(demo['examples']) > 2:
                        print(f'      • ... and {len(demo["examples"]) - 2} more')
        
        print('\n' + '=' * 50)
        print('💡 Usage: uv run cli smolagents --config "<configuration_name>"')
        print('   Example: uv run cli smolagents --config "Titanic"')
        
    except FileNotFoundError:
        print(f'❌ Configuration file not found: {config_file}')
    except yaml.YAMLError as e:
        print(f'❌ Error parsing YAML file: {e}')
    except Exception as e:
        print(f'❌ Error: {e}')


def get_available_config_names(config_type: str) -> list[str]:
    """Get list of available configuration names.
    
    Args:
        config_type: Type of configuration ('react_agent' or 'smolagents')
        
    Returns:
        List of available configuration names
    """
    config_files = {
        'react_agent': ('config/demos/react_agent.yaml', 'react_agent_demos'),
        'smolagents': ('config/demos/codeact_agent.yaml', 'codeact_agent_demos')
    }
    
    if config_type not in config_files:
        return []
    
    config_file, section_key = config_files[config_type]
    config_path = Path(config_file)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if section_key in config:
            return [demo.get('name', f'Demo {i}') for i, demo in enumerate(config[section_key], 1)]
        return []
        
    except Exception:
        return []