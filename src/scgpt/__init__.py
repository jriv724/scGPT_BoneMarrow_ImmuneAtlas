"""scGPT Bone Marrow Immune Atlas pipelines.

This package extends the upstream ``scgpt`` package namespace so local
``scgpt.pipelines`` modules can coexist with upstream ``scgpt.tasks``.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
