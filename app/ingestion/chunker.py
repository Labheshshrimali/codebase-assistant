"""Chunk source files into semantic units (functions / classes) using
tree-sitter, instead of naive fixed-size line windows."""
from dataclasses import dataclass
from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)
CHUNK_NODE_TYPES = {"function_definition", "class_definition"}


@dataclass
class Chunk:
    repo_relative_path: str
    start_line: int
    end_line: int
    node_type: str
    name: str
    code: str

    @property
    def id(self) -> str:
        return f"{self.repo_relative_path}:{self.start_line}-{self.end_line}"

    @property
    def citation(self) -> str:
        return f"{self.repo_relative_path}#L{self.start_line}-L{self.end_line}"


def _node_name(node, source):
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return "<anonymous>"


def _make_chunk(node, source, rel_path, name, node_type=None):
    code = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return Chunk(rel_path, node.start_point[0] + 1, node.end_point[0] + 1, node_type or node.type, name, code)


def _class_header_chunk(class_node, source, rel_path, class_name, first_method_node):
    end_byte = first_method_node.start_byte if first_method_node else class_node.end_byte
    end_row = first_method_node.start_point[0] if first_method_node else class_node.end_point[0]
    code = source[class_node.start_byte:end_byte].decode("utf-8", errors="replace")
    return Chunk(rel_path, class_node.start_point[0] + 1, max(end_row, class_node.start_point[0] + 1), "class_header", class_name, code)


def chunk_file(file_path: Path, repo_root: Path) -> list[Chunk]:
    source = file_path.read_bytes()
    tree = _parser.parse(source)
    rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    chunks = []

    def collect_methods(block_node):
        methods = []
        for c in block_node.children:
            if c.type == "function_definition":
                methods.append(c)
            elif c.type == "decorated_definition":
                for gc in c.children:
                    if gc.type == "function_definition":
                        methods.append(gc)
        return methods

    def walk(node, enclosing_class=None):
        if node.type == "function_definition":
            name = _node_name(node, source)
            full_name = f"{enclosing_class}.{name}" if enclosing_class else name
            chunks.append(_make_chunk(node, source, rel_path, full_name))
            return
        if node.type == "class_definition":
            class_name = _node_name(node, source)
            body_node = next((c for c in node.children if c.type == "block"), None)
            method_nodes = collect_methods(body_node) if body_node else []
            if method_nodes:
                first_method = min(method_nodes, key=lambda n: n.start_byte)
                chunks.append(_class_header_chunk(node, source, rel_path, class_name, first_method))
                walk(body_node, enclosing_class=class_name)
            else:
                chunks.append(_make_chunk(node, source, rel_path, class_name))
            return
        for child in node.children:
            walk(child, enclosing_class=enclosing_class)

    walk(tree.root_node)
    if not chunks and source.strip():
        chunks.append(Chunk(rel_path, 1, source.count(b"\n") + 1, "module", file_path.stem, source.decode("utf-8", errors="replace")))
    return chunks


def chunk_repo(repo_path: Path, files: list[Path]) -> list[Chunk]:
    all_chunks = []
    for f in files:
        try:
            all_chunks.extend(chunk_file(f, repo_path))
        except Exception as e:
            print(f"skip {f}: {e}")
    return all_chunks
