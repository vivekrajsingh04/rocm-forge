"""
ROCm Forge — Deployment Generator Agent
Generates Dockerfile, deployment scripts, requirements.txt, and
environment setup for AMD Developer Cloud / ROCm deployment.
"""
from knowledge.templates import (
    DOCKERFILE_TEMPLATE,
    DOCKER_COMPOSE_TEMPLATE,
    DEPLOY_SCRIPT_TEMPLATE,
    REQUIREMENTS_TEMPLATE,
    ENV_SETUP_TEMPLATE,
    AMD_CLOUD_GUIDE,
)
from knowledge.cuda_mappings import PIP_PACKAGE_MAPPINGS
import re


class DeployerAgent:
    """
    Deployment Generator Agent — Creates all deployment artifacts needed
    to run the migrated code on AMD Developer Cloud.
    """
    
    def __init__(self):
        self.name = "Deployment Generator Agent"
        self.trace_log = []
    
    def generate_all(self, code: str, analysis_result, refactored_code: str) -> dict:
        """
        Generate all deployment artifacts.
        Returns dict with: dockerfile, docker_compose, deploy_script, requirements, env_setup, cloud_guide
        """
        self.trace_log = []
        
        # Extract info from analysis
        packages = self._extract_packages(code, analysis_result)
        entry_point = self._detect_entry_point(code)
        
        self.trace_log.append(f"📋 Detected entry point: {entry_point}")
        self.trace_log.append(f"📦 Found {len(packages)} packages to migrate")
        
        # Generate each artifact
        dockerfile = self._generate_dockerfile(packages, entry_point)
        self.trace_log.append("🐳 Generated ROCm Dockerfile")
        
        docker_compose = self._generate_docker_compose()
        self.trace_log.append("🐳 Generated Docker Compose file")
        
        deploy_script = self._generate_deploy_script(packages, entry_point)
        self.trace_log.append("🚀 Generated deployment script")
        
        requirements = self._generate_requirements(packages)
        self.trace_log.append("📦 Generated ROCm requirements.txt")
        
        env_setup = self._generate_env_setup()
        self.trace_log.append("🔧 Generated environment setup script")
        
        cloud_guide = AMD_CLOUD_GUIDE
        self.trace_log.append("☁️ Attached AMD Developer Cloud guide")
        
        self.trace_log.append(f"✅ All deployment artifacts generated successfully")
        
        return {
            "dockerfile": dockerfile,
            "docker_compose": docker_compose,
            "deploy_script": deploy_script,
            "requirements": requirements,
            "env_setup": env_setup,
            "cloud_guide": cloud_guide,
        }
    
    def _extract_packages(self, code: str, analysis) -> list:
        """Extract Python packages from the code."""
        packages = set()
        
        # Parse import statements
        import_patterns = [
            r'^import\s+([\w]+)',
            r'^from\s+([\w]+)',
        ]
        
        for line in code.split("\n"):
            stripped = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, stripped)
                if match:
                    pkg = match.group(1)
                    # Map module names to pip package names
                    pkg_map = {
                        "torch": "torch",
                        "torchvision": "torchvision",
                        "torchaudio": "torchaudio",
                        "transformers": "transformers",
                        "datasets": "datasets",
                        "peft": "peft",
                        "vllm": "vllm",
                        "accelerate": "accelerate",
                        "bitsandbytes": "bitsandbytes",
                        "safetensors": "safetensors",
                        "sentencepiece": "sentencepiece",
                        "tokenizers": "tokenizers",
                        "wandb": "wandb",
                        "numpy": "numpy",
                        "pandas": "pandas",
                        "scipy": "scipy",
                        "sklearn": "scikit-learn",
                        "matplotlib": "matplotlib",
                        "tqdm": "tqdm",
                        "PIL": "Pillow",
                        "cv2": "opencv-python",
                        "gradio": "gradio",
                        "streamlit": "streamlit",
                        "fastapi": "fastapi",
                        "uvicorn": "uvicorn",
                        "flask": "flask",
                    }
                    if pkg in pkg_map:
                        packages.add(pkg_map[pkg])
        
        return sorted(packages)
    
    def _detect_entry_point(self, code: str) -> str:
        """Detect the main entry point script name."""
        if '__name__ == "__main__"' in code or "__name__ == '__main__'" in code:
            return "main.py"
        if "def main()" in code:
            return "main.py"
        if "app.run" in code or "uvicorn.run" in code:
            return "server.py"
        if "st.set_page_config" in code or "streamlit" in code.lower():
            return "app.py"
        return "main.py"
    
    def _generate_dockerfile(self, packages: list, entry_point: str) -> str:
        """Generate a ROCm Dockerfile."""
        # Build pip install commands
        pip_lines = []
        rocm_pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        remaining_packages = [p for p in packages if p not in rocm_pytorch_packages]
        
        # Handle special packages
        final_packages = []
        for pkg in remaining_packages:
            if pkg in PIP_PACKAGE_MAPPINGS:
                info = PIP_PACKAGE_MAPPINGS[pkg]
                if info["action"] == "remove":
                    continue
                elif info["action"] == "replace" and "install" in info:
                    pip_lines.append(f"RUN {info['install']}")
                    continue
            final_packages.append(pkg)
        
        if final_packages:
            pip_lines.append(f"RUN pip3 install --no-cache-dir {' '.join(final_packages)}")
        
        pip_install_str = "\n".join(pip_lines) if pip_lines else "# No additional dependencies"
        
        return DOCKERFILE_TEMPLATE.format(
            rocm_version="6.2",
            gfx_version="11.0.0",
            pip_install_commands=pip_install_str,
            entry_point=entry_point,
        )
    
    def _generate_docker_compose(self) -> str:
        """Generate Docker Compose file."""
        return DOCKER_COMPOSE_TEMPLATE.format(
            gfx_version="11.0.0",
            port="8000",
        )
    
    def _generate_deploy_script(self, packages: list, entry_point: str) -> str:
        """Generate deployment bash script."""
        pip_lines = []
        rocm_pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        remaining = [p for p in packages if p not in rocm_pytorch_packages]
        
        final = []
        for pkg in remaining:
            if pkg in PIP_PACKAGE_MAPPINGS:
                info = PIP_PACKAGE_MAPPINGS[pkg]
                if info["action"] == "remove":
                    continue
                elif info["action"] == "replace" and "install" in info:
                    pip_lines.append(info["install"])
                    continue
            final.append(pkg)
        
        if final:
            pip_lines.append(f"pip install {' '.join(final)}")
        
        pip_str = "\n".join(pip_lines) if pip_lines else "echo '  No additional packages needed'"
        
        return DEPLOY_SCRIPT_TEMPLATE.format(
            rocm_version="6.2",
            pip_install_commands=pip_str,
            entry_point=entry_point,
        )
    
    def _generate_requirements(self, packages: list) -> str:
        """Generate ROCm-compatible requirements.txt."""
        lines = []
        rocm_pytorch_packages = {"torch", "torchvision", "torchaudio"}
        
        for pkg in packages:
            if pkg in rocm_pytorch_packages:
                lines.append(f"# {pkg} — Install separately: pip install {pkg} --index-url https://download.pytorch.org/whl/rocm6.2")
                continue
            
            if pkg in PIP_PACKAGE_MAPPINGS:
                info = PIP_PACKAGE_MAPPINGS[pkg]
                if info["action"] == "remove":
                    lines.append(f"# REMOVED: {pkg} — {info['note']}")
                elif info["action"] == "replace":
                    replacement = info.get("replacement", pkg)
                    install = info.get("install", "")
                    lines.append(f"# {pkg} → {replacement}")
                    lines.append(f"# Install: {install}")
                elif info["action"] == "warning":
                    lines.append(f"{pkg}  # ⚠️ {info.get('note', 'Check ROCm compat')}")
                else:
                    lines.append(pkg)
            else:
                lines.append(pkg)
        
        return REQUIREMENTS_TEMPLATE.format(requirements="\n".join(lines))
    
    def _generate_env_setup(self) -> str:
        """Generate environment setup script."""
        return ENV_SETUP_TEMPLATE.format(
            gpu_ids="0",
            gfx_version="11.0.0",
        )
