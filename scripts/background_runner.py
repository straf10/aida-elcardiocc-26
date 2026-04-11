#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

def start_training(config_path):
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    pid_file = Path(".training.pid")
    
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        try:
            os.kill(int(pid), 0)
            print(f"Error: Training might already be running with PID {pid}.")
            print("Run 'python scripts/background_runner.py stop' first, or remove .training.pid")
            sys.exit(1)
        except OSError:
            # Process doesn't exist, safe to overwrite
            pass

    print(f"Starting detached training with config: {config_path}")
    print(f"Logs will be written to: {log_file}")
    
    # Use double fork to daemonize
    try:
        pid = os.fork()
        if pid > 0:
            # First parent exit
            time.sleep(1) # Wait a second to allow child to write PID file
            if pid_file.exists():
                print(f"Started successfully. PID: {pid_file.read_text().strip()}")
            sys.exit(0)
    except OSError as e:
        print(f"Fork #1 failed: {e.errno} ({e.strerror})", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.setsid()
    
    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Second parent exit
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e.errno} ({e.strerror})", file=sys.stderr)
        sys.exit(1)

    # We are now a daemon
    # Write the PID file
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    
    with open(log_file, "a") as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())
    
    with open(os.devnull, "r") as f:
        os.dup2(f.fileno(), sys.stdin.fileno())

    # Execute the training script
    try:
        from src.xlm_r.train import main
        # Modify sys.argv so argparse in train.py works correctly
        sys.argv = ["train.py", "--config", config_path]
        main()
    except Exception as e:
        import traceback
        print(f"Training failed with exception:\n{traceback.format_exc()}", file=sys.stderr)
    finally:
        if pid_file.exists():
            pid_file.unlink()

def get_status():
    pid_file = Path(".training.pid")
    if not pid_file.exists():
        print("No training process is currently running (no .training.pid file).")
        return
        
    pid = pid_file.read_text().strip()
    try:
        os.kill(int(pid), 0)
        print(f"Training process is RUNNING with PID: {pid}")
        
        # Try to find the most recent log file
        log_dir = Path("outputs/logs")
        if log_dir.exists():
            logs = sorted(log_dir.glob("train_*.log"))
            if logs:
                print(f"Latest log file: {logs[-1]}")
    except OSError:
        print(f"Process {pid} is DEAD, but .training.pid still exists. Cleaning up.")
        pid_file.unlink()

def stop_training():
    pid_file = Path(".training.pid")
    if not pid_file.exists():
        print("No training process is currently running.")
        return
        
    pid = pid_file.read_text().strip()
    try:
        import signal
        os.kill(int(pid), signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid}.")
        
        # Wait a bit
        time.sleep(2)
        try:
            os.kill(int(pid), 0)
            print(f"Process {pid} did not terminate. Sending SIGKILL...")
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            print(f"Process {pid} terminated successfully.")
            
        pid_file.unlink()
    except OSError:
        print(f"Process {pid} was not running. Cleaning up pid file.")
        pid_file.unlink()

def tail_logs(lines=50, follow=False):
    log_dir = Path("outputs/logs")
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist.")
        return
        
    logs = sorted(log_dir.glob("train_*.log"))
    if not logs:
        print("No log files found.")
        return
        
    latest_log = logs[-1]
    print(f"Showing logs from {latest_log}:")
    
    cmd = ["tail", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(latest_log))
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background runner for ELCardioCC training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start training in the background")
    start_parser.add_argument("--config", required=True, help="Path to config YAML file")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check if training is running")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the running training process")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View training logs")
    logs_parser.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_training(args.config)
    elif args.command == "status":
        get_status()
    elif args.command == "stop":
        stop_training()
    elif args.command == "logs":
        tail_logs(args.lines, args.follow)
    else:
        parser.print_help()