import logging
import os
import sys
import tomllib
import torch


class ConfigSection:
    """Class enabling dot-notation access for TOML sections."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigSection(v))
            else:
                setattr(self, k, v)


class AppConfig:
    """Singleton class managing application configuration and paths."""

    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(base_dir, "../../.."))

        if config_path is None:
            config_path = os.path.join(project_root, "app.toml")

        # --- Define default values ---
        self.dataset_dir = os.path.join(project_root, "dataset")
        self.log_dir = os.path.join(project_root, "log")
        self.space_dir = os.path.join(project_root, "space")
        
        self.max_train_samples = -1
        self.max_test_samples = -1
        self.patch_size = 9
        self.boundary_retention_percent = 20.0  # Default retention rate (top 20%)
        self.logger_name = "v1ai"

        self.similarity_threshold = 0.85
        self.training_trigger_count = 3000
        self.max_samples_per_class = 300

        # --- Load TOML file ---
        if os.path.exists(config_path):
            try:
                with open(config_path, "rb") as f:
                    data = tomllib.load(f)

                for section_name, section_values in data.items():
                    if isinstance(section_values, dict):
                        # Convert each section into a ConfigSection and set it
                        setattr(self, section_name, ConfigSection(section_values))

                        # Reflect paths and parameters within the [app] section
                        if section_name == "app":
                            for key in ["dataset_dir", "log_dir", "space_dir"]:
                                if key in section_values:
                                    setattr(
                                        self,
                                        key,
                                        os.path.normpath(os.path.join(project_root, section_values[key]))
                                    )
                            if "patch_size" in section_values:
                                self.patch_size = int(section_values["patch_size"])
                            if "boundary_retention_percent" in section_values:
                                self.boundary_retention_percent = float(section_values["boundary_retention_percent"])
                    else:
                        setattr(self, section_name, section_values)

            except Exception as e:
                print(f"[Warning] Failed to load app.toml: {e}. Falling back to default values.")

        # --- Initialize device and logger ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("[ERROR] GPU (CUDA) is not available. A GPU is required.")
            sys.exit(1)

        self._setup_logger()
        self._log_gpu_info()

    def _setup_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.space_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "app.log")

        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y/%m/%d %H:%M:%S",
            )

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    @property
    def logger(self):
        return logging.getLogger(self.logger_name)

    def _log_gpu_info(self):
        log = self.logger
        log.info("--- PyTorch GPU Diagnostic ---")
        current_idx = torch.cuda.current_device()
        log.info(f"Current CUDA Device Index: {current_idx}")
        log.info(f"CUDA Device Name: {torch.cuda.get_device_name(current_idx)}")
        log.info("-----------------------------")