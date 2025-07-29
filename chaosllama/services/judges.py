from pathlib import Path
def extract_text(node):
    if hasattr(node, 'children'):
        return ''.join(extract_text(child) for child in node.children)
    return str(node)

class JudgeService():
    """
    The purpose of this class is to manage the judges that will be used to evaluate the AI generated responses.
    """
    def __init__(self, guidelines_path="./customer_guidelines.md", global_guidelines: dict=None, scorers=None):
        self.guidelines = None
        self.guidelines_path = guidelines_path
        self.global_guidelines = global_guidelines
        self.judges = {}
        self.ai_judges = None
        self.scorers = scorers

    def load_guidelines(self, path: Path = None) -> str:
        self.guidelines = open(self.guidelines_path if self.guidelines_path else path, "r").read()
        return self

    def extract_text(self, node):
        if hasattr(node, 'children'):
            return ''.join(extract_text(child) for child in node.children)
        return str(node)

    def parse_guidelines(self, markdown: str = None):
        if not self.guidelines:
            self.load_guidelines()

        else:
            parsed = marko.parse(self.guidelines)
            result = {}
            current_heading = None
            for node in parsed.children:
                if isinstance(node, Heading):
                    current_heading = extract_text(node).strip(':')  # remove trailing colon
                    result[current_heading] = []
                elif isinstance(node, List) and current_heading:
                    for item in node.children:
                        bullet = extract_text(item).strip()
                        result[current_heading].append(bullet)

            self.guidelines = result
            return self, result

    def create_judges_from_guidelines(self, guidelines: list = None):
        pass