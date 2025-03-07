import torch
import torch.nn as nn
import re
from typing import Dict, List, Tuple

class RewardModelFromLLM(nn.Module):
    """
    Convert LLM reward into a trainable PyTorch model
    """
    def __init__(self, reward_code: str):
        super().__init__()
        
        # Extract scalar variables and create parameters
        self.scalar_vars = self._extract_scalar_vars(reward_code)
        self.param_dict = nn.ParameterDict({
            name: nn.Parameter(torch.tensor([float(value)], requires_grad=True))
            for name, value in self.scalar_vars
        })
        
        # Store original code and placeholder for modified code
        self.original_code = reward_code
        self.modified_code = None
        
    def _extract_scalar_vars(self, code_string: str) -> List[Tuple[str, float]]:
        """Extract scalar variables from the code string using regex"""
        scalar_pattern = r'(?:^|\n)\s*([\w\d_]+)\s*=\s*([+-]?\d*\.?\d+)(?:\s*#[^\n]*)?'
        return re.findall(scalar_pattern, code_string)
    
    def get_modified_reward_function(self) -> str:
        """
        Generate a new version of the reward function that uses the learned parameters
        """
        modified_code = self.original_code
        
        # Update function signature to accept parameters dictionary
        modified_code = re.sub(
            r'def compute_reward\((.*?)\)', 
            r'def compute_reward(params: Dict[str, torch.Tensor], \1)', 
            modified_code
        )
        
        # Replace each scalar definition with parameter reference
        for name, _ in self.scalar_vars:
            modified_code = re.sub(
                fr'(\s+){name}\s*=\s*[+-]?\d*\.?\d+(?:\s*#[^\n]*)?',
                fr'\1{name} = params["{name}"].item()',
                modified_code
            )
        
        self.modified_code = modified_code
        return modified_code
    
    def forward(self, *args, **kwargs):
        """
        This is a placeholder. The actual reward computation happens in the modified function.
        """
        raise NotImplementedError("This model doesn't compute rewards directly. Use the generated function instead.")
    
    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """Return the current parameters as a dictionary"""
        return {name: param for name, param in self.param_dict.items()}

def extract_and_parameterize_reward(code_string: str) -> Tuple[RewardModelFromLLM, str]:
    """
    Extract scalar parameters from reward function code and create a parameterized version
    
    Args:
        code_string: String containing the compute_reward function
        
    Returns:
        model: RewardModelFromLLM instance with trainable parameters
        new_code: Modified code with parameterized reward function
    """
    model = RewardModelFromLLM(code_string)
    new_code = model.get_modified_reward_function()
    
    # Debug output
    print("\n=== ORIGINAL REWARD FUNCTION ===")
    print(code_string)
    print("\n=== PARAMETERIZED REWARD FUNCTION ===")
    print(new_code)
    print("\n=== EXTRACTED PARAMETERS ===")
    for name, value in model.scalar_vars:
        print(f"{name} = {value}")
    
    return model, new_code

def apply_to_env_file(env_file_path: str, output_file_path: str) -> RewardModelFromLLM:
    """
    Extract the compute_reward function from an env file, parameterize it, and save to a new file
    
    Args:
        env_file_path: Path to the environment file containing the reward function
        output_file_path: Path to save the modified file
        
    Returns:
        model: RewardModelFromLLM instance with trainable parameters
    """
    # Read the env file
    with open(env_file_path, 'r') as f:
        content = f.read()
    
    # Extract the compute_reward function
    reward_func_pattern = r'@torch\.jit\.script\ndef compute_reward\(.*?\):.*?return.*?\n}'
    match = re.search(reward_func_pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Could not find compute_reward function in the file")
    
    reward_func = match.group(0)
    
    # Create parameterized model and get modified code
    model, modified_reward_func = extract_and_parameterize_reward(reward_func)
    
    # Replace the original reward function with the modified one
    modified_content = content.replace(reward_func, modified_reward_func)
    
    # Add imports for Dict if not already present
    if 'from typing import Dict' not in modified_content:
        modified_content = 'from typing import Dict\n' + modified_content
    
    # Write the modified content to the output file
    with open(output_file_path, 'w') as f:
        f.write(modified_content)
    
    return model

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and parameterize reward function')
    parser.add_argument('--input', type=str, required=True, help='Path to input env file')
    parser.add_argument('--output', type=str, required=True, help='Path to output env file')
    
    args = parser.parse_args()
    
    model = apply_to_env_file(args.input, args.output)
    print(f"Extracted parameters: {model.scalar_vars}")
    print(f"Modified reward function saved to {args.output}") 