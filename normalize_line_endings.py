import os
import fnmatch

# Read .gitignore patterns
ignore_patterns = []
if os.path.exists('.gitignore'):
    with open('.gitignore', 'r') as f:
        for line in f:
            pattern = line.strip()
            if pattern and not pattern.startswith('#'):
                ignore_patterns.append(pattern)

def is_ignored(path):
    for pattern in ignore_patterns:
        # Handle directory ignore
        if pattern.endswith('/') and os.path.commonpath([os.path.abspath(path), os.path.abspath(pattern.rstrip('/'))]) == os.path.abspath(pattern.rstrip('/')):
            return True
        # Handle file pattern ignore
        if fnmatch.fnmatch(path, pattern):
            return True
    return False

# Walk through all files in the current directory and subdirectories
for root, dirs, files in os.walk('.'):
    # Remove ignored directories from walk
    dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
    for file in files:
        file_path = os.path.join(root, file)
        if is_ignored(file_path):
            continue
        # Skip .git directory and binary files
        if '.git' in root or file.endswith(('.png', '.jpg', '.jpeg', '.exe', '.dll', '.so', '.pyc', '.zip', '.tar', '.gz', '.ico')):
            continue
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            # Only process text files
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                continue
            # Replace CRLF and CR with LF
            new_text = text.replace('\r\n', '\n').replace('\r', '\n')
            if new_text != text:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(new_text)
                print(f'Normalized: {file_path}')
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
