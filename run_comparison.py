#!/usr/bin/env python3
"""
Run SPARKS vs Control Models Comparison
This script runs all models with the same parameters for fair comparison
"""

import subprocess
import sys
import time
import os

def run_model(script_name, args, model_name):
    """Run a model script and return the result"""
    print(f"\n🚀 Running {model_name}...")
    print(f"Command: python {script_name} {' '.join(args)}")
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_name] + args, 
                              capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {model_name} completed successfully in {end_time - start_time:.1f}s")
            return True, result.stdout
        else:
            print(f"❌ {model_name} failed with error:")
            print(result.stderr)
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} timed out after 1 hour")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {model_name} failed with exception: {e}")
        return False, str(e)

def main():
    """Run the complete comparison study"""
    print("🎯 SPARKS vs Control Models Comparison Study")
    print("=" * 60)
    
    # Common parameters for all models
    common_args = [
        "--n_epochs", "20",
        "--test_period", "5", 
        "--batch_size", "32",
        "--lr", "0.001"
    ]
    
    models = [
        ("monkey_reaching.py", ["--target_type", "hand_pos"] + common_args, "SPARKS (Hebbian Attention)"),
        ("controls/monkey_conventional_attention.py", common_args, "Conventional Attention"),
        ("controls/monkey_vae.py", ["--enc_type", "linear"] + common_args, "Linear VAE"),
        ("controls/monkey_vae.py", ["--enc_type", "rnn"] + common_args, "RNN VAE")
    ]
    
    results = {}
    
    for script, args, name in models:
        success, output = run_model(script, args, name)
        results[name] = {"success": success, "output": output}
        
        if not success:
            print(f"⚠️  Skipping {name} due to failure")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON STUDY SUMMARY")
    print("=" * 60)
    
    successful_models = [name for name, result in results.items() if result["success"]]
    failed_models = [name for name, result in results.items() if not result["success"]]
    
    print(f"✅ Successful models: {len(successful_models)}")
    for model in successful_models:
        print(f"   - {model}")
    
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\n📁 Results saved in: {os.getcwd()}/results/")
    print("\n💡 To analyze results:")
    print("   1. Check the results/ directory for model outputs")
    print("   2. Compare test_acc.npy files for performance metrics")
    print("   3. Use the saved model weights for further analysis")

if __name__ == "__main__":
    main()
