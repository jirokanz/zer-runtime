
import ast

BLOCKED_IMPORTS = {"os", "subprocess", "socket", "shutil", "sys", "ctypes", "fcntl", "posix", "pty", "signal"}

def validate_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in BLOCKED_IMPORTS:
                violations.append(f"Blocked import: {node.module}")
    return violations
