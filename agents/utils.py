def log_node(fn):
    def wrapper(self, state):
        self.logger.debug(f"[Node:start] {fn.__name__}")
        out = fn(self, state)
        self.logger.debug(f"[Node:end] {fn.__name__}")
        return out
    return wrapper
