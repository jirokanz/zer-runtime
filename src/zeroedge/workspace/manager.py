class WorkspaceManager:
    def __init__(self, root="/tmp/workspace"):
        self.root = root
    def create_workspace(self, task_id):
        import os
        path = f"{self.root}/{task_id}"
        os.makedirs(path, exist_ok=True)
        return path
