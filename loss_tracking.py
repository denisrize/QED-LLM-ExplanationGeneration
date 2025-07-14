import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback
import os
import json

class LossTrackingCallback(TrainerCallback):
    """Custom callback to track and plot training/evaluation losses."""
    
    def __init__(self, output_dir="./training_plots", plot_every_steps=100):
        self.output_dir = output_dir
        self.plot_every_steps = plot_every_steps
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics."""
        if logs is None:
            return
            
        # Track training loss
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)
            
        # Track evaluation loss
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)
            
        # Plot every N steps
        if state.global_step % self.plot_every_steps == 0 and len(self.train_losses) > 1:
            self.plot_losses(state.global_step)
            
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.plot_losses(state.global_step, final=True)
        self.save_loss_data()
        
    def plot_losses(self, current_step, final=False):
        """Create and save loss plots."""
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        if self.train_losses:
            plt.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        # Plot 2: Training vs Evaluation Loss
        plt.subplot(1, 2, 2)
        if self.train_losses:
            plt.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss', alpha=0.7)
        if self.eval_losses:
            plt.plot(self.eval_steps, self.eval_losses, 'r-', label='Evaluation Loss', alpha=0.7)
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Evaluation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        suffix = "_final" if final else f"_step_{current_step}"
        plt.savefig(os.path.join(self.output_dir, f"loss_plot{suffix}.png"), dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
    def save_loss_data(self):
        """Save loss data to JSON for later analysis."""
        data = {
            "train_steps": self.train_steps,
            "train_losses": self.train_losses,
            "eval_steps": self.eval_steps,
            "eval_losses": self.eval_losses
        }
        
        with open(os.path.join(self.output_dir, "loss_data.json"), "w") as f:
            json.dump(data, f, indent=2)
            
    def get_loss_data(self):
        """Return current loss data."""
        return {
            "train_steps": self.train_steps.copy(),
            "train_losses": self.train_losses.copy(),
            "eval_steps": self.eval_steps.copy(),
            "eval_losses": self.eval_losses.copy()
        }


def plot_saved_losses(loss_data_path, output_path="loss_analysis.png"):
    """Plot losses from saved JSON data."""
    with open(loss_data_path, "r") as f:
        data = json.load(f)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 3, 1)
    if data["train_losses"]:
        plt.plot(data["train_steps"], data["train_losses"], 'b-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
    # Plot 2: Evaluation Loss
    plt.subplot(1, 3, 2)
    if data["eval_losses"]:
        plt.plot(data["eval_steps"], data["eval_losses"], 'r-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss')
        plt.grid(True, alpha=0.3)
        
    # Plot 3: Combined
    plt.subplot(1, 3, 3)
    if data["train_losses"]:
        plt.plot(data["train_steps"], data["train_losses"], 'b-', label='Training', linewidth=2, alpha=0.8)
    if data["eval_losses"]:
        plt.plot(data["eval_steps"], data["eval_losses"], 'r-', label='Evaluation', linewidth=2, alpha=0.8)
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Evaluation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    if data["train_losses"]:
        print(f"Training Loss: Start={data['train_losses'][0]:.4f}, End={data['train_losses'][-1]:.4f}")
        print(f"Training Loss: Min={min(data['train_losses']):.4f}, Max={max(data['train_losses']):.4f}")
    
    if data["eval_losses"]:
        print(f"Eval Loss: Min={min(data['eval_losses']):.4f}, Max={max(data['eval_losses']):.4f}")
        
    return data 