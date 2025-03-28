import re
import ast
import logging
import random
import torch
import os
import shutil

def extract_scalar_parameters(reward_function_code):
    """
    Extract all scalar numerical parameters from a reward function code.
    
    Args:
        reward_function_code (str): The string containing the reward function code
        
    Returns:
        dict: A dictionary mapping parameter names to their values
    """
    parameters = {}
    
    # Parse the function code into an AST
    try:
        parsed = ast.parse(reward_function_code)
        
        # Define a visitor class to find all numerical constants
        class ScalarVisitor(ast.NodeVisitor):
            def __init__(self):
                self.constants = {}
                self.current_assignment = None
            
            def visit_Assign(self, node):
                # Check for assignments that might be parameters
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                        # Direct assignment of a scalar
                        self.constants[target_name] = node.value.value
                    elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub) and isinstance(node.value.operand, ast.Constant):
                        # Handle negative numbers
                        self.constants[target_name] = -node.value.operand.value
                
                # Continue visiting other nodes
                self.generic_visit(node)
        
        # Apply the visitor to the parsed code
        visitor = ScalarVisitor()
        visitor.visit(parsed)
        parameters = visitor.constants
        
    except SyntaxError as e:
        logging.warning(f"Syntax error when parsing reward function: {e}")
        # Fallback to regex-based extraction for basic cases
        # Find all assignments of the form variable = number
        assignments = re.findall(r'(\w+)\s*=\s*(-?\d+\.?\d*)', reward_function_code)
        for name, value in assignments:
            try:
                if '.' in value:
                    parameters[name] = float(value)
                else:
                    parameters[name] = int(value)
            except ValueError:
                continue
    
    return parameters

def analyze_reward_components(reward_dict):
    """
    Analyze the components of a reward dictionary to provide insights.
    
    Args:
        reward_dict (dict): Dictionary mapping reward component names to their values
        
    Returns:
        dict: Analysis results including total contribution and percentage
    """
    analysis = {}
    total = sum(reward_dict.values())
    
    for component, value in reward_dict.items():
        analysis[component] = {
            'value': value,
            'percentage': (value / total * 100) if total != 0 else 0
        }
    
    return analysis

def randomize_parameters(parameters, min_val=15, max_val=20):
    """
    Takes a dictionary of parameters and replaces their values with random
    numbers between min_val and max_val.
    
    Args:
        parameters (dict): Dictionary of parameter names and their values
        min_val (float): Minimum value for randomization (default: 15)
        max_val (float): Maximum value for randomization (default: 20)
        
    Returns:
        dict: Dictionary with the same keys but randomized values
    """
    randomized_params = {}
    
    for param_name, param_value in parameters.items():
        # Generate a random float between min_val and max_val
        random_value = random.uniform(min_val, max_val)
        
        # Preserve type (int or float) of original parameter
        if isinstance(param_value, int):
            random_value = int(random_value)
            
        randomized_params[param_name] = random_value
    
    return randomized_params

def update_reward_function_with_parameters(reward_function_code, new_parameters):
    """
    Updates the reward function code with new parameter values.
    
    Args:
        reward_function_code (str): The original reward function code
        new_parameters (dict): Dictionary of parameter names and their new values
        
    Returns:
        str: Updated reward function code with new parameter values
    """
    updated_code = reward_function_code
    
    for param_name, param_value in new_parameters.items():
        # Create pattern to find the parameter assignment
        pattern = rf'({param_name}\s*=\s*)([+-]?\d+\.?\d*)'
        
        # Format the new value based on its type
        if isinstance(param_value, int):
            replacement = r'\g<1>' + f"{param_value}"
        else:
            replacement = r'\g<1>' + f"{param_value:.6f}"
        
        # Replace the parameter in the code
        updated_code = re.sub(pattern, replacement, updated_code)
    
    return updated_code

def create_tensor_parameters(parameters, device="cuda:0"):
    """
    Converts a dictionary of scalar parameters to a dictionary of PyTorch tensors.
    
    Args:
        parameters (dict): Dictionary of parameter names and their scalar values
        device (str): Device to place the tensors on (default: "cuda:0")
        
    Returns:
        dict: Dictionary with the same keys but with tensor values
    """
    tensor_params = {}
    
    for param_name, param_value in parameters.items():
        # Convert each scalar to a PyTorch tensor
        if isinstance(param_value, (int, float)):
            tensor_params[param_name] = torch.tensor(param_value, device=device)
        else:
            # If the value is already a tensor, ensure it's on the right device
            if hasattr(param_value, 'to'):
                tensor_params[param_name] = param_value.to(device)
            else:
                # Skip parameters that can't be converted to tensors
                continue
    
    return tensor_params

def llm_reward_to_nn_module(reward_code, output_file=None, base_dir=None):
    """
    Converts an LLM-generated reward function to a PyTorch nn.Module class
    and writes it to prototype_test.py
    
    Args:
        reward_code (str): The reward function code from LLM
        output_file (str): Path to write the modified prototype_test.py file 
        base_dir (str): Base directory where prototype_test.py is located
    """
    import re
    import torch
    import os
    
    # Determine file paths
    if base_dir is None:
        prototype_path = 'prototype_test.py'
    else:
        prototype_path = os.path.join(base_dir, 'prototype_test.py')
    
    if output_file is None:
        output_file = prototype_path
        
    print(f"Using prototype template from: {prototype_path}")
    print(f"Writing output to: {output_file}")
    
    try:
        # Clean up the reward code - remove @torch.jit.script if present
        reward_code = re.sub(r'@torch\.jit\.script\s*', '', reward_code)
        
        # Extract function signature, removing return type annotations
        function_match = re.search(r'def\s+([^(]+)\s*\(([^)]*)\)', reward_code)
        if not function_match:
            print("Failed to extract function signature")
            return False
            
        function_name = function_match.group(1).strip()
        args_text = function_match.group(2)
        
        # Clean up argument list - remove return type annotations and keep only parameter names
        args_clean = []
        for arg in args_text.split(','):
            # Extract just the parameter name before any type annotation
            param_name = arg.split(':')[0].strip()
            if param_name:
                args_clean.append(param_name)
        
        function_args = args_clean
        
        print(f"Extracted function: {function_name} with args: {function_args}")
        
        # Extract scalar parameters - numbers assigned to variables
        scalar_params = {}
        scalar_pattern = r'(\w+)\s*=\s*([+-]?\d+\.?\d*)'
        for line in reward_code.split('\n'):
            line = line.strip()
            matches = re.findall(scalar_pattern, line)
            for match in matches:
                param_name, param_value = match
                try:
                    # Only include if it's actually a number
                    float(param_value)  # Will raise ValueError if not a number
                    scalar_params[param_name] = param_value
                except ValueError:
                    continue
        
        print(f"Extracted {len(scalar_params)} scalar parameters: {scalar_params}")
        
        # Extract the function body
        body_start = reward_code.find(':', reward_code.find('def'))
        if body_start == -1:
            print("Failed to find function body start")
            return False
            
        # Get everything after the colon
        body_text = reward_code[body_start+1:].strip()
        
        print(f"Function body length: {len(body_text)} characters")
        
        # Start building the new RewardFunction class
        nn_module_code = """
# LLM's reward function transformed to nn.Module
class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()
"""
        
        # Add parameters as nn.Parameters in __init__
        for param_name, param_value in scalar_params.items():
            nn_module_code += f"        self.{param_name} = nn.Parameter(torch.tensor([{param_value}], requires_grad=True))\n"
        
        # Add forward method without any return type annotations
        nn_module_code += f"\n    def forward(self, {', '.join(function_args)}):\n"
        
        # Process the function body
        for line in body_text.split('\n'):
            stripped = line.strip()
            # If the line is assigning a known scalar param, skip it
            if any(re.match(rf'^\s*{p}\s*=', stripped) for p in scalar_params):
                continue
            
            # Otherwise, replace references and add it to forward
            for param_name in scalar_params:
                line = re.sub(r'\b' + param_name + r'\b', f"self.{param_name}", line)
                # Fix dimension parameter errors: replace self.dim=-1 with dim=-1
                line = re.sub(r'self\.dim\s*=\s*(-?\d+)', r'dim=\1', line)
            nn_module_code += f"        {line}\n"
        
        print("Generated new RewardFunction class successfully")
        
        # Get existing file content
        with open(prototype_path, 'r') as f:
            content = f.read()
        
        # Replace the RewardFunction class
        pattern = r'class RewardFunction\(nn\.Module\):.*?(?=\n\n\w|$)'
        new_content = re.sub(pattern, nn_module_code.strip(), content, flags=re.DOTALL)
        
        # Verify replacement worked
        if new_content == content:
            print("WARNING: Replacement pattern didn't match anything")
            # Try a simpler pattern
            pattern = r'class RewardFunction\(nn\.Module\):.*?\n\n'
            new_content = re.sub(pattern, nn_module_code.strip() + "\n\n", content, flags=re.DOTALL)
            
            # If still no match, try to find the class definition and replace from there to the next class
            if new_content == content:
                class_start = content.find("class RewardFunction(nn.Module):")
                if class_start != -1:
                    next_section = content.find("\n\n", class_start)
                    if next_section != -1:
                        new_content = content[:class_start] + nn_module_code + content[next_section:]
                    else:
                        new_content = content[:class_start] + nn_module_code
        
        # Write updated content
        with open(output_file, 'w') as f:
            f.write(new_content)
        
        print(f"Updated RewardFunction class in {output_file}")
        
        # Verification step - check if our code was actually written
        with open(output_file, 'r') as f:
            verification = f.read()
            
        if "def forward(" in verification and any(param in verification for param in scalar_params.keys()):
            print("Verification passed - RewardFunction class was updated successfully")
            return True
        else:
            print("WARNING: Verification failed - RewardFunction might not have been updated correctly")
            return False
            
    except Exception as e:
        print(f"Error converting reward function to nn.Module: {e}")
        import traceback
        traceback.print_exc()
        return False 