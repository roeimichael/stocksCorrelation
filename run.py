#!/usr/bin/env python3
"""
Convenience wrapper for src.run module.
Allows running: python run.py [command]
Instead of: python -m src.run [command]
"""

if __name__ == "__main__":
    from src.run import main
    main()
