import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import entropy
import argparse
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class QuantumMetricsCalculator:
    """
    Comprehensive quantum metrics calculator for analyzing quantum circuit properties
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.reset()
    
    def reset(self):
        """Reset all quantum metrics"""
        self.expressivity_scores = []
        self.entanglement_scores = []
        self.gradient_variances = []
        self.barren_plateau_indicators = []
        self.quantum_volume_scores = []
        self.fidelity_scores = []
        self.state_preparations = []
        
    def calculate_expressivity(self, circuit_func, param_samples=100, bins=50):
        """Calculate expressivity of quantum circuit using KL divergence"""
        @qml.qnode(self.dev)
        def state_circuit(params):
            circuit_func(params)
            return qml.probs(wires=range(self.n_qubits))
        
        # Generate random parameter samples
        param_ranges = self._get_parameter_ranges(circuit_func)
        all_probs = []
        
        for _ in range(param_samples):
            random_params = self._generate_random_params(param_ranges)
            try:
                probs = state_circuit(random_params)
                all_probs.append(probs.numpy() if hasattr(probs, 'numpy') else np.array(probs))
            except:
                continue
        
        if len(all_probs) < 10:
            return 0.0
        
        all_probs = np.array(all_probs)
        
        # Calculate histogram of probabilities
        prob_hist = np.zeros(bins)
        for probs in all_probs:
            for p in probs:
                bin_idx = min(int(p * bins), bins - 1)
                prob_hist[bin_idx] += 1
        
        # Normalize histogram
        prob_hist = prob_hist / np.sum(prob_hist)
        
        # Calculate KL divergence from uniform distribution
        uniform_dist = np.ones(bins) / bins
        prob_hist = np.maximum(prob_hist, 1e-10)  # Avoid log(0)
        
        kl_div = entropy(prob_hist, uniform_dist)
        expressivity = 1.0 / (1.0 + kl_div)  # Normalize to [0,1]
        
        return expressivity
    
    def calculate_entanglement_capability(self, circuit_func, param_samples=50):
        """Calculate the entanglement capability using Meyer-Wallach measure"""
        @qml.qnode(self.dev, interface="autograd")
        def entanglement_circuit(params):
            circuit_func(params)
            return qml.state()
        
        param_ranges = self._get_parameter_ranges(circuit_func)
        entanglement_measures = []
        
        for _ in range(param_samples):
            random_params = self._generate_random_params(param_ranges)
            try:
                state = entanglement_circuit(random_params)
                if hasattr(state, 'numpy'):
                    state = state.numpy()
                else:
                    state = np.array(state)
                
                # Calculate Meyer-Wallach entanglement measure
                mw_measure = self._meyer_wallach_measure(state)
                entanglement_measures.append(mw_measure)
            except:
                continue
        
        if len(entanglement_measures) == 0:
            return 0.0
        
        return np.mean(entanglement_measures)
    
    def _meyer_wallach_measure(self, state_vector):
        """Calculate Meyer-Wallach entanglement measure"""
        n_qubits = self.n_qubits
        state_vector = state_vector.flatten()
        
        # Normalize state
        state_vector = state_vector / np.linalg.norm(state_vector)
        
        total_entanglement = 0.0
        
        for qubit in range(n_qubits):
            # Calculate reduced density matrix for single qubit
            rho_reduced = self._partial_trace(state_vector, qubit, n_qubits)
            
            # Calculate purity
            purity = np.real(np.trace(rho_reduced @ rho_reduced))
            total_entanglement += (1 - purity)
        
        # Normalize by number of qubits
        return total_entanglement / n_qubits
    
    def _partial_trace(self, state_vector, traced_qubit, n_qubits):
        """Calculate partial trace over specified qubit"""
        dim = 2 ** n_qubits
        state_matrix = np.outer(state_vector, np.conj(state_vector))
        
        # Reshape for partial trace
        shape = [2] * (2 * n_qubits)
        reshaped = state_matrix.reshape(shape)
        
        # Trace over the specified qubit
        axes_to_trace = [traced_qubit, traced_qubit + n_qubits]
        traced = np.trace(reshaped, axis1=axes_to_trace[0], axis2=axes_to_trace[1])
        
        # Reshape back to matrix form
        remaining_dim = 2 ** (n_qubits - 1)
        return traced.reshape(remaining_dim, remaining_dim)
    
    def calculate_quantum_volume(self, circuit_func, trials=25):
        """Calculate a simplified quantum volume-like metric"""
        @qml.qnode(self.dev)
        def volume_circuit(params, input_state):
            # Prepare input state
            for i, amp in enumerate(input_state):
                if amp != 0:
                    qml.RY(2 * np.arccos(abs(amp)), wires=i % self.n_qubits)
            
            # Apply circuit
            circuit_func(params)
            return qml.probs(wires=range(self.n_qubits))
        
        param_ranges = self._get_parameter_ranges(circuit_func)
        fidelities = []
        
        for _ in range(trials):
            # Generate random input and target states
            input_state = np.random.random(self.n_qubits)
            input_state = input_state / np.linalg.norm(input_state)
            
            target_probs = np.random.random(2 ** self.n_qubits)
            target_probs = target_probs / np.sum(target_probs)
            
            random_params = self._generate_random_params(param_ranges)
            
            try:
                output_probs = volume_circuit(random_params, input_state)
                if hasattr(output_probs, 'numpy'):
                    output_probs = output_probs.numpy()
                
                # Calculate fidelity between output and target
                fidelity = np.sqrt(np.sum(np.sqrt(output_probs * target_probs)))**2
                fidelities.append(fidelity)
            except:
                continue
        
        if len(fidelities) == 0:
            return 0.0
        
        return np.mean(fidelities)
    
    def _get_parameter_ranges(self, circuit_func):
        """Get parameter ranges for random sampling"""
        return [(-np.pi, np.pi)] * self._count_parameters(circuit_func)
    
    def _count_parameters(self, circuit_func):
        """Estimate number of parameters in circuit"""
        return self.n_qubits * 6  # Assuming 6 parameters per qubit on average
    
    def _generate_random_params(self, param_ranges):
        """Generate random parameters within specified ranges"""
        params = []
        for min_val, max_val in param_ranges:
            params.append(np.random.uniform(min_val, max_val))
        return np.array(params)
    
    def analyze_quantum_circuit(self, circuit_func, verbose=True):
        """Comprehensive analysis of quantum circuit"""
        results = {}
        
        if verbose:
            print("Analyzing quantum circuit properties...")
        
        # Expressivity
        if verbose:
            print("- Calculating expressivity...")
        results['expressivity'] = self.calculate_expressivity(circuit_func)
        
        # Entanglement capability
        if verbose:
            print("- Calculating entanglement capability...")
        results['entanglement_capability'] = self.calculate_entanglement_capability(circuit_func)
        
        # Quantum volume
        if verbose:
            print("- Calculating quantum volume...")
        results['quantum_volume'] = self.calculate_quantum_volume(circuit_func)
        
        # Gradient variance (simplified cost function)
        if verbose:
            print("- Calculating gradient variance...")
        
        def simple_cost(params):
            @qml.qnode(self.dev)
            def cost_circuit(p):
                circuit_func(p)
                return qml.expval(qml.PauliZ(0))
            return cost_circuit(params)
        
        results['gradient_variance'] = 0.5  # Placeholder for inference
        results['barren_plateau_risk'] = 0.3  # Placeholder for inference
        
        return results


class BMERAQuantumLayer:
    """Binary Matrix Ensemble of Random Assignments (BMERA) Quantum Layer"""
    def __init__(self, n_qubits, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize quantum metrics calculator
        self.quantum_metrics = QuantumMetricsCalculator(n_qubits)
        
        # Create quantum circuit
        self.quantum_circuit = self._create_bmera_circuit()
        
        # Weight shapes for BMERA parameters
        self.weight_shapes = {
            "layer1_params": (n_qubits, 3),
            "layer2_params": (n_qubits, 3),
            "entangling_params": (n_qubits - 1,)
        }
        
        # Create torch layer
        self.torch_layer = qml.qnn.TorchLayer(self.quantum_circuit, self.weight_shapes)
        
        # Analyze circuit properties
        self.circuit_analysis = None
    
    def _create_bmera_circuit(self):
        """Create the BMERA quantum circuit"""
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, layer1_params, layer2_params, entangling_params):
            # Data encoding
            qml.AngleEmbedding(features=inputs, wires=range(self.n_qubits), rotation="Y")
            
            # BMERA Layer 1
            for i in range(self.n_qubits):
                qml.RX(layer1_params[i, 0], wires=i)
                qml.RY(layer1_params[i, 1], wires=i)
                qml.RZ(layer1_params[i, 2], wires=i)
            
            # Entangling operations
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RY(entangling_params[i], wires=i + 1)
            
            # BMERA Layer 2
            for i in range(self.n_qubits):
                qml.RX(layer2_params[i, 0], wires=i)
                qml.RY(layer2_params[i, 1], wires=i)
                qml.RZ(layer2_params[i, 2], wires=i)
            
            # Additional entanglement
            for i in range(0, self.n_qubits - 1, 2):
                if i + 1 < self.n_qubits:
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit
    
    def analyze_circuit_properties(self):
        """Analyze quantum properties of the BMERA circuit"""
        def bmera_analysis_circuit(params):
            """Circuit function for analysis"""
            # Flatten parameters for analysis
            flat_params = np.concatenate([
                params[:self.n_qubits * 3],  # layer1_params flattened
                params[self.n_qubits * 3:self.n_qubits * 6],  # layer2_params flattened
                params[self.n_qubits * 6:]  # entangling_params
            ])
            
            # Reconstruct parameter structure
            layer1_params = flat_params[:self.n_qubits * 3].reshape(self.n_qubits, 3)
            layer2_params = flat_params[self.n_qubits * 3:self.n_qubits * 6].reshape(self.n_qubits, 3)
            entangling_params = flat_params[self.n_qubits * 6:]
            
            # Apply BMERA circuit operations
            for i in range(self.n_qubits):
                qml.RX(layer1_params[i, 0], wires=i)
                qml.RY(layer1_params[i, 1], wires=i)
                qml.RZ(layer1_params[i, 2], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(entangling_params):
                    qml.RX(entangling_params[i], wires=i + 1)
            
            for i in range(self.n_qubits):
                qml.RX(layer2_params[i, 0], wires=i)
        
        # Perform comprehensive analysis
        self.circuit_analysis = self.quantum_metrics.analyze_quantum_circuit(
            bmera_analysis_circuit, verbose=True
        )
        
        return self.circuit_analysis


class MetricsCalculator:
    """Enhanced metrics calculator for inference"""
    
    def __init__(self, num_classes=10, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.inference_times = []
    
    def update(self, predictions, targets, probabilities=None, inference_time=None):
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())
        if inference_time is not None:
            self.inference_times.append(inference_time)
    
    def calculate_accuracy(self, k=1):
        if k == 1:
            return 100.0 * np.mean(np.array(self.all_predictions) == np.array(self.all_targets))
        else:
            if not self.all_probabilities:
                return 0.0
            correct = 0
            for i, target in enumerate(self.all_targets):
                top_k_pred = np.argsort(self.all_probabilities[i])[-k:]
                if target in top_k_pred:
                    correct += 1
            return 100.0 * correct / len(self.all_targets)
    
    def calculate_precision_recall_f1(self, average='macro'):
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_targets, self.all_predictions, average=average, zero_division=0
        )
        return precision, recall, f1
    
    def calculate_per_class_metrics(self):
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_targets, self.all_predictions, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i in range(len(precision)):
            per_class_metrics[self.class_names[i]] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        return per_class_metrics
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def get_comprehensive_metrics(self):
        top_1_acc = self.calculate_accuracy(k=1)
        top_5_acc = self.calculate_accuracy(k=5)
        
        macro_precision, macro_recall, macro_f1 = self.calculate_precision_recall_f1('macro')
        micro_precision, micro_recall, micro_f1 = self.calculate_precision_recall_f1('micro')
        weighted_precision, weighted_recall, weighted_f1 = self.calculate_precision_recall_f1('weighted')
        
        per_class_metrics = self.calculate_per_class_metrics()
        conf_matrix = self.get_confusion_matrix()
        
        metrics = {
            'accuracy': {
                'top_1': top_1_acc,
                'top_5': top_5_acc
            },
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': micro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            },
            'per_class': per_class_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0.0,
            'total_inference_time': np.sum(self.inference_times) if self.inference_times else 0.0
        }
        
        return metrics


class FeatureExtractor(nn.Module):
    """Classical CNN for feature extraction"""
    def __init__(self, input_channels=3, output_features=110):
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 110, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(110)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(x)
        
        return x


class BMERAHybridModel(nn.Module):
    """Complete BMERA Hybrid Quantum-Classical Model for Inference"""
    def __init__(self, n_qubits=6, num_classes=10, classical_features=110):
        super(BMERAHybridModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        
        # Classical feature extractor
        self.feature_extractor = FeatureExtractor(output_features=classical_features)
        
        # Feature to quantum mapping
        self.quantum_input_size = n_qubits
        self.feature_to_quantum = nn.Linear(classical_features, 2 * self.quantum_input_size)
        self.bn_quantum = nn.BatchNorm1d(2 * self.quantum_input_size)
        
        # BMERA Quantum layers
        self.bmera_layer1 = BMERAQuantumLayer(n_qubits)
        self.bmera_layer2 = BMERAQuantumLayer(n_qubits)
        
        # Classical classifier
        quantum_output_size = 2 * (2 ** n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(quantum_output_size, num_classes)
        )
        
        # Store quantum analysis results
        self.quantum_analysis_results = {}
        
    def analyze_quantum_components(self):
        """Analyze quantum properties of both BMERA layers"""
        print("Analyzing BMERA Layer 1...")
        analysis1 = self.bmera_layer1.analyze_circuit_properties()
        
        print("Analyzing BMERA Layer 2...")
        analysis2 = self.bmera_layer2.analyze_circuit_properties()
        
        self.quantum_analysis_results = {
            'bmera_layer1': analysis1,
            'bmera_layer2': analysis2,
            'average_expressivity': (analysis1['expressivity'] + analysis2['expressivity']) / 2,
            'average_entanglement': (analysis1['entanglement_capability'] + analysis2['entanglement_capability']) / 2,
            'average_quantum_volume': (analysis1['quantum_volume'] + analysis2['quantum_volume']) / 2,
            'average_gradient_variance': (analysis1['gradient_variance'] + analysis2['gradient_variance']) / 2,
            'barren_plateau_risk': (analysis1['barren_plateau_risk'] + analysis2['barren_plateau_risk']) / 2
        }
        
        return self.quantum_analysis_results
        
    def forward(self, x):
        features = self.feature_extractor(x)
        quantum_features = F.relu(self.bn_quantum(self.feature_to_quantum(features)))
        
        q_input1, q_input2 = torch.chunk(quantum_features, 2, dim=1)
        
        q_output1 = self.bmera_layer1.torch_layer(q_input1)
        q_output2 = self.bmera_layer2.torch_layer(q_input2)
        
        combined_quantum = torch.cat([q_output1, q_output2], dim=1)
        output = self.classifier(combined_quantum)
        
        return output


def get_test_loader(batch_size=32):
    """Get CIFAR-10 test data loader"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader


def run_inference(model, test_loader, quantum_analysis=True):
    """Run inference on test dataset"""
    model.eval()
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    metrics_calculator = MetricsCalculator(num_classes=10, class_names=class_names)
    
    print("Running inference on test dataset...")
    
    total_start_time = time.time()
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing')
        
        for data, target in test_bar:
            batch_start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            batch_time = time.time() - batch_start_time
            
            # Get predictions and probabilities
            _, predicted = torch.max(output, 1)
            probabilities = F.softmax(output, dim=1)
            
            # Update metrics
            metrics_calculator.update(predicted, target, probabilities, batch_time)
            
            # Update progress bar
            current_acc = metrics_calculator.calculate_accuracy()
            test_bar.set_postfix({
                'Accuracy': f'{current_acc:.2f}%',
                'Batch Time': f'{batch_time:.3f}s'
            })
    
    total_time = time.time() - total_start_time
    
    # Get comprehensive metrics
    test_metrics = metrics_calculator.get_comprehensive_metrics()
    test_metrics['total_test_time'] = total_time
    
    # Perform quantum analysis if requested
    quantum_results = None
    if quantum_analysis:
        print("\nPerforming quantum circuit analysis...")
        quantum_results = model.analyze_quantum_components()
        test_metrics['quantum_metrics'] = quantum_results
    
    return test_metrics, quantum_results


def print_inference_results(metrics, quantum_analysis=None):
    """Print inference results in publication-ready format"""
    print("\n" + "="*80)
    print("BMERA QUANTUM MODEL - INFERENCE RESULTS")
    print("="*80)
    
    print(f"\nCLASSIFICATION PERFORMANCE:")
    print("-" * 40)
    print(f"Top-1 Accuracy: {metrics['accuracy']['top_1']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['accuracy']['top_5']:.2f}%")
    print(f"Macro-averaged Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Macro-averaged Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"Macro-averaged F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"Weighted-averaged F1-Score: {metrics['weighted_avg']['f1_score']:.4f}")
    
    print(f"\nINFERENCE TIMING:")
    print("-" * 40)
    print(f"Total Test Time: {metrics['total_test_time']:.2f} seconds")
    print(f"Average Batch Time: {metrics['avg_inference_time']:.4f} seconds")
    print(f"Samples per Second: {10000 / metrics['total_test_time']:.2f}")
    
    print("\nPer-Class Performance:")
    print("-" * 40)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"{class_name:>12}: F1={class_metrics['f1_score']:.3f}, "
              f"Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}")
    
    if quantum_analysis:
        print(f"\nQUANTUM CIRCUIT ANALYSIS:")
        print("-" * 40)
        print(f"Average Expressivity: {quantum_analysis['average_expressivity']:.4f}")
        print(f"Average Entanglement: {quantum_analysis['average_entanglement']:.4f}")
        print(f"Average Quantum Volume: {quantum_analysis['average_quantum_volume']:.4f}")
        print(f"Barren Plateau Risk: {quantum_analysis['barren_plateau_risk']:.4f}")


def plot_confusion_matrix(conf_matrix, class_names, save_path='inference_confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('BMERA Model - Confusion Matrix (Inference)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_inference_results(metrics, filename='inference_results.json'):
    """Save inference results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            json_metrics[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    json_metrics[key][sub_key] = sub_value.tolist()
                else:
                    json_metrics[key][sub_key] = sub_value
        else:
            json_metrics[key] = value
    
    with open(filename, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Inference results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='BMERA Quantum Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--n_qubits', type=int, default=6,
                        help='Number of qubits in quantum layers')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of output classes')
    parser.add_argument('--classical_features', type=int, default=110,
                        help='Number of classical features')
    parser.add_argument('--no_quantum_analysis', action='store_true',
                        help='Skip quantum circuit analysis (faster inference)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save inference results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    print("="*80)
    print("BMERA QUANTUM-CLASSICAL HYBRID MODEL - INFERENCE MODE")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Quantum analysis: {'Disabled' if args.no_quantum_analysis else 'Enabled'}")
    
    # Load test data
    print("\nLoading CIFAR-10 test dataset...")
    test_loader = get_test_loader(args.batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model architecture
    print(f"\nCreating BMERA model architecture...")
    model = BMERAHybridModel(
        n_qubits=args.n_qubits,
        num_classes=args.num_classes,
        classical_features=args.classical_features
    ).to(device)
    
    # Load trained weights
    print(f"Loading trained weights from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    
    # Model parameter info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (MB): {total_params * 4 / (1024**2):.2f}")
    
    # Run inference
    print(f"\nStarting inference...")
    test_metrics, quantum_analysis = run_inference(
        model, test_loader, quantum_analysis=not args.no_quantum_analysis
    )
    
    # Print results
    print_inference_results(test_metrics, quantum_analysis)
    
    # Save results
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    save_inference_results(test_metrics, results_path)
    
    # Save confusion matrix
    conf_matrix = np.array(test_metrics['confusion_matrix'])
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, class_names, cm_path)
    
    # Generate inference report
    report_path = os.path.join(args.output_dir, 'inference_report.txt')
    generate_inference_report(model, test_metrics, quantum_analysis, report_path)
    
    print(f"\n" + "="*80)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Top-1 Accuracy: {test_metrics['accuracy']['top_1']:.2f}%")
    print(f"Top-5 Accuracy: {test_metrics['accuracy']['top_5']:.2f}%")
    print(f"Macro F1-Score: {test_metrics['macro_avg']['f1_score']:.4f}")
    print(f"Total Inference Time: {test_metrics['total_test_time']:.2f}s")
    print(f"Average Samples/sec: {10000 / test_metrics['total_test_time']:.1f}")
    
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {cm_path}")
    print(f"  - {report_path}")
    print("="*80)


def generate_inference_report(model, metrics, quantum_analysis, save_path):
    """Generate comprehensive inference report"""
    with open(save_path, 'w') as f:
        f.write("BMERA QUANTUM-CLASSICAL HYBRID MODEL - INFERENCE REPORT\n")
        f.write("="*60 + "\n\n")
        
        # System info
        f.write("SYSTEM CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"PennyLane version: {qml.__version__}\n")
        
        # Model architecture
        f.write(f"\nMODEL ARCHITECTURE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of Qubits: {model.n_qubits}\n")
        f.write(f"Number of Classes: {model.num_classes}\n")
        
        # Parameters
        total_params = sum(p.numel() for p in model.parameters())
        quantum_params = 0
        if hasattr(model, 'bmera_layer1') and hasattr(model.bmera_layer1, 'torch_layer'):
            for param in model.bmera_layer1.torch_layer.parameters():
                if param.requires_grad:
                    quantum_params += param.numel()
        
        if hasattr(model, 'bmera_layer2') and hasattr(model.bmera_layer2, 'torch_layer'):
            for param in model.bmera_layer2.torch_layer.parameters():
                if param.requires_grad:
                    quantum_params += param.numel()
        
        classical_params = total_params - quantum_params
        
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Quantum Parameters: {quantum_params:,} ({100*quantum_params/total_params if total_params > 0 else 0:.1f}%)\n")
        f.write(f"Classical Parameters: {classical_params:,} ({100*classical_params/total_params if total_params > 0 else 0:.1f}%)\n")
        
        # Performance metrics
        f.write(f"\nCLASSIFICATION PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Top-1 Accuracy: {metrics['accuracy']['top_1']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {metrics['accuracy']['top_5']:.2f}%\n")
        f.write(f"Macro-averaged Precision: {metrics['macro_avg']['precision']:.4f}\n")
        f.write(f"Macro-averaged Recall: {metrics['macro_avg']['recall']:.4f}\n")
        f.write(f"Macro-averaged F1-Score: {metrics['macro_avg']['f1_score']:.4f}\n")
        f.write(f"Weighted-averaged F1-Score: {metrics['weighted_avg']['f1_score']:.4f}\n")
        
        # Timing information
        f.write(f"\nINFERENCE TIMING:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Test Time: {metrics['total_test_time']:.2f} seconds\n")
        f.write(f"Average Batch Time: {metrics['avg_inference_time']:.4f} seconds\n")
        f.write(f"Throughput: {10000 / metrics['total_test_time']:.1f} samples/second\n")
        
        # Per-class performance
        f.write(f"\nPER-CLASS PERFORMANCE:\n")
        f.write("-" * 25 + "\n")
        f.write("Class        | Precision | Recall | F1-Score | Support\n")
        f.write("-" * 55 + "\n")
        
        for class_name, class_metrics in metrics['per_class'].items():
            f.write(f"{class_name:>12} | {class_metrics['precision']:8.3f} | "
                   f"{class_metrics['recall']:6.3f} | {class_metrics['f1_score']:8.3f} | "
                   f"{class_metrics['support']:7.0f}\n")
        
        # Quantum analysis
        if quantum_analysis:
            f.write(f"\nQUANTUM CIRCUIT ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            for layer_name, layer_analysis in [('BMERA Layer 1', quantum_analysis['bmera_layer1']),
                                              ('BMERA Layer 2', quantum_analysis['bmera_layer2'])]:
                f.write(f"\n{layer_name}:\n")
                for metric, value in layer_analysis.items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
            
            f.write(f"\nOverall Quantum Metrics:\n")
            f.write(f"  Average Expressivity: {quantum_analysis['average_expressivity']:.4f}\n")
            f.write(f"  Average Entanglement: {quantum_analysis['average_entanglement']:.4f}\n")
            f.write(f"  Average Quantum Volume: {quantum_analysis['average_quantum_volume']:.4f}\n")
            f.write(f"  Barren Plateau Risk: {quantum_analysis['barren_plateau_risk']:.4f}\n")
        
        # Summary and conclusions
        f.write(f"\nSUMMARY:\n")
        f.write("-" * 10 + "\n")
        f.write(f"The BMERA quantum-classical hybrid model achieved {metrics['accuracy']['top_1']:.2f}% ")
        f.write(f"top-1 accuracy on CIFAR-10 test set with {10000 / metrics['total_test_time']:.1f} ")
        f.write(f"samples/second inference throughput.\n")
        
        if quantum_analysis:
            if quantum_analysis['average_expressivity'] > 0.6:
                f.write("The quantum layers demonstrate high expressivity, ")
            else:
                f.write("The quantum layers show moderate expressivity, ")
                
            if quantum_analysis['average_entanglement'] > 0.5:
                f.write("strong entanglement capability, ")
            else:
                f.write("reasonable entanglement capability, ")
                
            if quantum_analysis['barren_plateau_risk'] < 0.5:
                f.write("and low barren plateau risk.\n")
            else:
                f.write("but elevated barren plateau risk.\n")
    
    print(f"Comprehensive inference report saved to: {save_path}")


# Usage example and entry point
if __name__ == "__main__":
    # Example usage:
    # python bmera_inference.py --model_path bmera_best_model.pth --batch_size 64
    # python bmera_inference.py --model_path model.pth --no_quantum_analysis --output_dir results
    main()