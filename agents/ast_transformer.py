"""
ROCm Forge — AST-Level Python Transformer
Uses Python's ast module to perform tree-level code transformation,
going beyond regex to understand code STRUCTURE.

This is the critical differentiator from hipify and other string-based tools.
"""
import ast
import re
from typing import List, Tuple, Dict


class CUDANodeVisitor(ast.NodeVisitor):
    """
    AST visitor that walks the Python syntax tree and detects
    CUDA-specific constructs at the structural level.
    """

    def __init__(self):
        self.findings: List[Dict] = []
        self.device_vars: Dict[str, int] = {}  # var_name -> line_no
        self.cuda_call_sites: List[Dict] = []
        self.env_mutations: List[Dict] = []
        self.string_literals_with_cuda: List[Dict] = []

    def visit_Assign(self, node: ast.Assign):
        """Detect device = torch.device('cuda:0') assignments."""
        if isinstance(node.value, ast.Call):
            func = node.value
            func_name = self._get_func_name(func)

            # torch.device("cuda:X") pattern
            if func_name == "torch.device" and func.args:
                arg = func.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if "cuda" in arg.value:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.device_vars[target.id] = node.lineno
                        self.findings.append({
                            "line": node.lineno,
                            "type": "device_assignment",
                            "original": f'torch.device("{arg.value}")',
                            "suggestion": f'torch.device("{arg.value}")  # Works on ROCm — PyTorch ROCm uses cuda namespace',
                            "severity": "info",
                        })

        # os.environ["CUDA_VISIBLE_DEVICES"] = ... pattern
        if isinstance(node.value, ast.Constant) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript):
                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    if "CUDA" in target.slice.value:
                        self.env_mutations.append({
                            "line": node.lineno,
                            "var": target.slice.value,
                            "type": "env_mutation",
                        })

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect CUDA API calls: .cuda(), torch.cuda.*, etc."""
        func_name = self._get_func_name(node)

        # .cuda() method calls
        if func_name and func_name.endswith(".cuda"):
            self.cuda_call_sites.append({
                "line": node.lineno,
                "call": func_name,
                "type": "cuda_method",
                "note": ".cuda() works on ROCm — maps to HIP backend",
            })

        # torch.cuda.is_available(), torch.cuda.device_count(), etc.
        if func_name and "torch.cuda" in func_name:
            self.cuda_call_sites.append({
                "line": node.lineno,
                "call": func_name,
                "type": "torch_cuda_api",
                "note": "torch.cuda.* APIs work on ROCm via HIP backend",
            })

        # load_inline / cpp_extension calls
        if func_name and ("load_inline" in func_name or "load" in func_name):
            for kw in node.keywords:
                if kw.arg == "cuda_sources":
                    self.findings.append({
                        "line": node.lineno,
                        "type": "inline_kernel",
                        "original": "cuda_sources=[...]",
                        "suggestion": "hip_sources=[...] (rename parameter for clarity)",
                        "severity": "review",
                    })

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect string literals containing CUDA code."""
        if isinstance(node.value, str) and len(node.value) > 50:
            cuda_indicators = [
                "cuda_runtime.h", "cudaMalloc", "cudaFree",
                "__global__", "cudaMemcpy", "cudaDeviceSynchronize",
                "<<<", ">>>",
            ]
            if any(ind in node.value for ind in cuda_indicators):
                self.string_literals_with_cuda.append({
                    "line": node.lineno,
                    "length": len(node.value),
                    "type": "embedded_cuda_kernel",
                    "note": "Embedded CUDA C/C++ kernel detected in string literal",
                })
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        """Detect `if not torch.cuda.is_available()` guards."""
        test_src = ast.dump(node.test)
        if "torch" in test_src and "cuda" in test_src:
            self.findings.append({
                "line": node.lineno,
                "type": "cuda_guard",
                "original": "CUDA availability check",
                "suggestion": "Works on ROCm — torch.cuda.is_available() returns True with HIP backend",
                "severity": "info",
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Detect imports from CUDA-specific modules."""
        if node.module and "cuda" in node.module.lower():
            self.findings.append({
                "line": node.lineno,
                "type": "cuda_import",
                "original": f"from {node.module} import ...",
                "suggestion": "Check ROCm compatibility for this import",
                "severity": "review",
            })
        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract dotted function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""


class ASTTransformer:
    """
    AST-level Python code transformer.
    Performs structural analysis beyond regex — understands scope, assignment,
    function calls, and string literal contexts.
    """

    def __init__(self):
        self.trace_log: List[str] = []
        self.ast_findings: List[Dict] = []

    def analyze(self, code: str) -> Tuple[List[Dict], List[str]]:
        """
        Perform AST-level analysis on Python code.
        Returns (findings, trace_log).
        Falls back gracefully if code can't be parsed.
        """
        self.trace_log = []
        self.ast_findings = []

        try:
            tree = ast.parse(code)
            self.trace_log.append("🌳 AST parse successful — performing tree-level analysis")
        except SyntaxError as e:
            self.trace_log.append(f"🌳 AST parse failed (line {e.lineno}) — code contains non-Python syntax, using regex fallback")
            return [], self.trace_log

        visitor = CUDANodeVisitor()
        visitor.visit(tree)

        # Aggregate findings
        self.ast_findings = visitor.findings.copy()

        # Report device variables
        for var, line in visitor.device_vars.items():
            self.ast_findings.append({
                "line": line,
                "type": "device_variable",
                "original": f'{var} = torch.device("cuda:...")',
                "suggestion": "Device variable tracked — all downstream .to(device) calls are AMD-safe",
                "severity": "info",
            })

        # Report embedded kernels
        for kernel in visitor.string_literals_with_cuda:
            self.ast_findings.append({
                "line": kernel["line"],
                "type": "embedded_kernel",
                "original": f"Inline CUDA kernel ({kernel['length']} chars)",
                "suggestion": "Embedded kernel requires hipify transformation of string contents",
                "severity": "critical",
            })

        # Report CUDA call sites
        for call in visitor.cuda_call_sites:
            self.ast_findings.append({
                "line": call["line"],
                "type": call["type"],
                "original": call["call"],
                "suggestion": call["note"],
                "severity": "info",
            })

        # Report env mutations
        for env in visitor.env_mutations:
            self.ast_findings.append({
                "line": env["line"],
                "type": "env_mutation",
                "original": f'os.environ["{env["var"]}"]',
                "suggestion": f'Needs migration to ROCm equivalent env var',
                "severity": "warning",
            })

        # Summary
        self.trace_log.append(
            f"🌳 AST Analysis: {len(visitor.device_vars)} device vars, "
            f"{len(visitor.cuda_call_sites)} CUDA calls, "
            f"{len(visitor.string_literals_with_cuda)} embedded kernels, "
            f"{len(visitor.env_mutations)} env mutations"
        )

        return self.ast_findings, self.trace_log
