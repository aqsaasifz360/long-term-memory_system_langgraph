"""Define utility functions for your graph."""

from typing import Optional

from langgraph.store.base import Item


def extract_memory_content(item):
    """Extract the actual content from a memory item with better error handling."""
    try:
        if hasattr(item, 'value') and item.value:
            value = item.value
            
            # Handle nested dictionary structures
            if isinstance(value, dict):
                # Try 'content' field first (most common)
                if 'content' in value:
                    content_field = value['content']
                    # Handle nested content
                    if isinstance(content_field, dict) and 'content' in content_field:
                        return str(content_field['content'])
                    elif content_field:
                        return str(content_field)
                
                # Try other common content fields
                for field in ['text', 'data', 'memory', 'description', 'info']:
                    if field in value and value[field]:
                        return str(value[field])
                
                # If it's a simple dict, return it as a string
                return str(value)
            
            # If value is already a string
            elif isinstance(value, str):
                return value
        
        # Fallback to string representation
        return str(item) if item else ""
        
    except Exception as e:
        print(f"DEBUG: Error extracting memory content from {item}: {e}")
        return str(item) if item else ""


def format_memories(memories: Optional[list[Item]]) -> str:
    """Format the user's memories with improved content extraction."""
    if not memories:
        return ""
    
    formatted_parts = []
    for m in memories:
        try:
            content = extract_memory_content(m)
            if content and content.strip() and content != "None":
                # Get memory type from namespace if available
                memory_type = "Memory"
                if hasattr(m, 'namespace') and m.namespace:
                    # Namespace format: (prefix, user_id, memory_type)
                    if len(m.namespace) > 2:
                        memory_type = m.namespace[-1]
                
                # Format with timestamp if available
                timestamp = ""
                if hasattr(m, 'updated_at') and m.updated_at:
                    timestamp = f" (Updated: {m.updated_at})"
                
                formatted_parts.append(f"[{memory_type}] {content}{timestamp}")
        except Exception as e:
            print(f"DEBUG: Error formatting memory {m}: {e}")
            continue
    
    if not formatted_parts:
        return ""
    
    formatted_memories = "\n".join(formatted_parts)
    return f"""## Memories

You have noted the following memorable information from previous interactions with the user:

<memories>
{formatted_memories}
</memories>
"""


def debug_memory_structure(item):
    """Debug helper to understand memory structure."""
    print(f"DEBUG: Memory item type: {type(item)}")
    if hasattr(item, '__dict__'):
        print(f"DEBUG: Memory item attributes: {list(item.__dict__.keys())}")
    if hasattr(item, 'value'):
        print(f"DEBUG: Memory value type: {type(item.value)}")
        print(f"DEBUG: Memory value: {str(item.value)[:200]}...")
    if hasattr(item, 'namespace'):
        print(f"DEBUG: Memory namespace: {item.namespace}")
    print("---")