"""Export the same shoulder-selected OC-SORT analysis as attention.py."""
from attention_app import ROOT, main


if __name__ == "__main__":
    raise SystemExit(main(export_default=ROOT / "output_demo.mp4"))
