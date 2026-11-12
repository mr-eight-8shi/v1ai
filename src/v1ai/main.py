import argparse
import sys
import shutil
import os

from .com import AppConfig
from .core import (
    BoundaryFilter,
    RegionalColorExtractor,
    ObjectExtractorFilter,  # Changed from BackgroundEraserFilter
    BoundaryVisualizer
)


def action_runner():
    """Decorator handling configuration loading and exception catching."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = AppConfig()
            print("\n" + "=" * 65)
            print("    ⚙️ [AppConfig] Query-Driven Boundary Filter System Initialized")
            print("=" * 65)
            print(f" • Workspace Directory : {config.space_dir}")
            print(f" • Log Directory       : {config.log_dir}")
            patch_size = getattr(getattr(config, "app", None), "patch_size", getattr(config, "patch_size", 9))
            print(f" • Patch Size          : {patch_size}x{patch_size}")
            print("=" * 65 + "\n")
            try:
                return func(config, *args, **kwargs)
            except KeyboardInterrupt:
                print("\n\n[ABORT] Interrupted by user.\n")
        return wrapper
    return decorator


@action_runner()
def run_train_color(config):
    """[Option 1] Train RegionalColorExtractor (Inside/Outside 10-Color)."""
    target_size = 32
    
    color_model = RegionalColorExtractor(config, target_size=target_size)
    print(f"--- [Training] RegionalColorExtractor ({target_size}x{target_size}) ---")
    color_model.train_model(epochs=20, num_samples=50000, patience=8)


@action_runner()
def run_train_boundary(config):
    """[Option 2] Train BoundaryFilter."""
    patch_size = getattr(getattr(config, "app", None), "patch_size", getattr(config, "patch_size", 9))
    
    filter_model = BoundaryFilter(config)
    print(f"--- [Training] BoundaryFilter ({patch_size}x{patch_size}) ---")
    filter_model.train_model(epochs=20, num_samples=200000, patience=5)


@action_runner()
def run_train_extractor(config):
    """[Option 3] Train ObjectExtractorFilter (Object & Boundary vs Background Classification)."""
    patch_size = getattr(getattr(config, "app", None), "patch_size", getattr(config, "patch_size", 9))
    print(f"--- [Training] ObjectExtractorFilter ({patch_size}x{patch_size}) ---")
    extractor_model = ObjectExtractorFilter(config)
    extractor_model.train_model(epochs=20, num_samples=200000, patience=10)


@action_runner()
def run_visualizer_pipeline(config):
    """[Option 4] Run BoundaryVisualizer for confidence mapping & object extraction."""
    patch_size = getattr(getattr(config, "app", None), "patch_size", getattr(config, "patch_size", 9))
    visualizer = BoundaryVisualizer(config, patch_size=patch_size)
    visualizer.visualize_sample_images()


@action_runner()
def run_init(config):
    """[Option 9] Initialize and clear the workspace directory."""
    if os.path.exists(config.space_dir):
        shutil.rmtree(config.space_dir)
        print(f"🧹 Workspace '{config.space_dir}' has been initialized.")


def interactive_menu():
    """Interactive CLI menu."""
    while True:
        print("\n" + "═" * 65)
        print("    🚀 Query-Driven MLP Boundary Filter & Color Extractor System")
        print("═" * 65)
        print(" [ Operations ]")
        print("    [1] Train RegionalColorExtractor (Inside r<=32 vs Outside r>32 Colors)")
        print("    [2] Train BoundaryFilter (Edge Confidence Classifier)")
        print("    [3] Train ObjectExtractorFilter (Object Extraction & Background Removal)")
        print("    [4] Run BoundaryVisualizer (Confidence Map & Segmentation)")
        print("-" * 65)
        print(" [ Others ]")
        print("    [9] Initialize Workspace (Reset)")
        print("    [q] Quit")
        print("═" * 65)

        try:
            choice = input("Enter choice: ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if choice in ["q", "quit"]:
            print("Exiting.")
            break
        elif choice == "1":
            run_train_color()
        elif choice == "2":
            run_train_boundary()
        elif choice == "3":
            run_train_extractor()
        elif choice == "4":
            run_visualizer_pipeline()
        elif choice in ["9", "99"]:
            confirm = input("⚠️ Initialize workspace? (y/n): ").strip().lower()
            if confirm == "y":
                run_init()
        else:
            print("[ERROR] Invalid choice.")


def cli():
    """CLI entry point for v1ai command."""
    parser = argparse.ArgumentParser(description="Query-Driven Boundary Filter CLI")
    parser.add_argument("action", nargs="?", help="Action to execute (train-color, train-boundary, train-extractor, visual, init)")
    args = parser.parse_args()

    action = args.action.lower() if args.action else ""
    
    if action in ["train-color", "1"]:
        run_train_color()
    elif action in ["train-boundary", "2"]:
        run_train_boundary()
    elif action in ["train-extractor", "train-object", "3"]:
        run_train_extractor()
    elif action in ["visual", "4", "5"]:
        run_visualizer_pipeline()
    elif action in ["init", "9", "99"]:
        run_init()
    else:
        interactive_menu()


if __name__ == "__main__":
    cli()