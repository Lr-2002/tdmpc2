import yaml

# Load the environment.yaml file
with open('./environment.yaml', 'r') as file:
    env = yaml.safe_load(file)

# Open requirements.txt for writing
with open('requirements.txt', 'w') as req_file:
    # Iterate through dependencies
    for dep in env['dependencies']:
        if isinstance(dep, str):  # Regular dependencies
            req_file.write(dep + '\n')
        elif isinstance(dep, dict) and 'pip' in dep:  # Pip dependencies
            for pip_dep in dep['pip']:
                req_file.write(pip_dep + '\n')

print("requirements.txt has been created.")

