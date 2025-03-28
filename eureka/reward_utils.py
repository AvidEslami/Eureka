import re
import ast
import logging
import random
import torch

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