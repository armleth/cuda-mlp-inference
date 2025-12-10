import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# --- 1. Define Model ---
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Standard forward pass
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_model(model, filename="mnist_weights.bin"):
    print(f"Exporting weights to {filename}...")
    state_dict = model.state_dict()
    with open(filename, "wb") as f:
        f.write(struct.pack("I", len(state_dict)))
        for key, tensor in state_dict.items():
            if "weight" in key:
                tensor = tensor.t()
            data = tensor.cpu().detach().numpy().flatten()
            f.write(struct.pack("I", data.size))
            f.write(data.tobytes())
    print("Model exported.")


def export_test_samples(loader, filename="mnist_samples.bin", num_samples=100):
    print(f"Exporting {num_samples} test samples to {filename}...")
    data_iter = iter(loader)
    # Ensure we get enough samples even if batch size is small
    images_list = []
    labels_list = []

    while len(images_list) * loader.batch_size < num_samples:
        imgs, lbls = next(data_iter)
        images_list.append(imgs)
        labels_list.append(lbls)

    # Concat to handle arbitrary batch sizes
    images = torch.cat(images_list)[:num_samples]
    labels = torch.cat(labels_list)[:num_samples]

    with open(filename, "wb") as f:
        f.write(struct.pack("I", num_samples))
        f.write(struct.pack("I", 28 * 28))
        for i in range(num_samples):
            img_flat = images[i].view(-1).numpy()
            f.write(img_flat.tobytes())
            label = int(labels[i])
            f.write(struct.pack("I", label))
    print("Samples exported.")


# --- Benchmarking Function ---
def benchmark_inference(model, device, dataset, num_samples=100):
    print(f"\nStarting PyTorch Inference on {num_samples} images...")

    # Create a loader with batch_size=1 to match C++ single-sample inference
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    # 1. Warmup
    # Run a few dummy passes to initialize CUDA context and kernels
    dummy = torch.randn(1, 28 * 28).to(device)
    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize()

    # 2. Setup Timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_fc1_time = 0.0
    total_relu_time = 0.0
    total_fc2_time = 0.0
    correct = 0

    # Prepare data iterator
    data_iter = iter(loader)

    # --- Start Global Timer (Wall Clock) ---
    global_start = time.time()

    with torch.no_grad():
        for i in range(num_samples):
            try:
                data, target = next(data_iter)
            except StopIteration:
                break

            # Move to device (part of loop overhead, but required for inference)
            # Flattening here to match the C++ input expectation
            x = data.view(-1, 28 * 28).to(device)
            target = target.item()

            # --- Layer 1: FC1 ---
            start_event.record()
            out1 = model.fc1(x)
            end_event.record()
            torch.cuda.synchronize()  # Strict sync to match C++ benchmark
            total_fc1_time += start_event.elapsed_time(end_event)

            # --- Layer 2: ReLU ---
            start_event.record()
            out2 = model.relu(out1)
            end_event.record()
            torch.cuda.synchronize()
            total_relu_time += start_event.elapsed_time(end_event)

            # --- Layer 3: FC2 ---
            start_event.record()
            out3 = model.fc2(out2)
            end_event.record()
            torch.cuda.synchronize()
            total_fc2_time += start_event.elapsed_time(end_event)

            # Prediction
            pred = out3.argmax(dim=1).item()
            if pred == target:
                correct += 1

            if i < 10:
                print(
                    f"Img {i} | Pred: {pred} | Actual: {target} "
                    f"{'[OK]' if pred == target else '[FAIL]'}"
                )

    # --- End Global Timer ---
    global_end = time.time()

    # Calculate stats
    global_duration_ms = (global_end - global_start) * 1000.0
    total_pure_gpu_ms = total_fc1_time + total_relu_time + total_fc2_time
    accuracy = 100.0 * correct / num_samples

    print("-" * 32)
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 32)
    print(f"BENCHMARK RESULTS ({num_samples} samples):")
    print(f"Total Global Time (incl. data loading/cpu): {global_duration_ms:.2f} ms")
    print(f"Total Pure GPU Inference Time:              {total_pure_gpu_ms:.2f} ms")
    print(
        f"Average Inference per sample:               {total_pure_gpu_ms / num_samples:.4f} ms"
    )
    print("\nLayer Breakdown (Average per pass):")
    print(f"  FC1 (784->128): {total_fc1_time / num_samples:.4f} ms")
    print(f"  ReLU:           {total_relu_time / num_samples:.4f} ms")
    print(f"  FC2 (128->10):  {total_fc2_time / num_samples:.4f} ms")


# --- Main Execution ---
if __name__ == "__main__":
    # Setup
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    # Batch size 100 for export, but we will use batch size 1 for benchmark
    test_loader_export = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MNIST_MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train
    print("Training...")
    model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Export
    export_model(model)
    export_test_samples(test_loader_export)

    # Benchmark
    # We pass the dataset directly so the benchmark can create its own batch_size=1 loader
    benchmark_inference(model, device, test_dataset, num_samples=100)
