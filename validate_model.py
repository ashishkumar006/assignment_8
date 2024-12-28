import nbformat
from nbconvert import PythonExporter
import importlib.util
import torch
import torch.nn as nn
import os

def extract_model_from_notebook(notebook_path, output_path):
    # Convert notebook to Python script
    with open(notebook_path, 'r') as nb_file:
        nb_content = nb_file.read()
    notebook = nbformat.reads(nb_content, as_version=4)
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(notebook)
    with open(output_path, 'w') as py_file:
        py_file.write(source)

def check_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params < 20000, total_params

def check_batch_norm(model):
    for layer in model.modules():
        if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return True
    return False

def check_dropout(model):
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            return True
    return False

def check_gap_or_fc(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.AvgPool2d)):
            return True
    return False

if __name__ == "__main__":
    notebook_path = "model_training.ipynb"
    script_path = "model_training.py"

    # Convert notebook to script
    extract_model_from_notebook(notebook_path, script_path)

    # Dynamically load the model from the script
    spec = importlib.util.spec_from_file_location("model_training", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Create model instance
    model = module.Net()
    
    # Perform validation checks
    param_check, total_params = check_parameters(model)
    batch_norm_check = check_batch_norm(model)
    dropout_check = check_dropout(model)
    gap_fc_check = check_gap_or_fc(model)

    # Print validation results
    print(f"Parameter Count Check (<20k): {param_check} (Total: {total_params})")
    print(f"BatchNorm Check: {batch_norm_check}")
    print(f"Dropout Check: {dropout_check}")
    print(f"GAP/FC Check: {gap_fc_check}")

    # Assert conditions
    assert param_check, f"Model has too many parameters: {total_params}"
    assert batch_norm_check, "Model does not use BatchNorm."
    assert dropout_check, "Model does not use Dropout."
    assert gap_fc_check, "Model does not use GAP or Fully Connected layer."

    print("All checks passed!")

    # Clean up
    os.remove(script_path)
