import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from scipy.stats import entropy
from scipy.linalg import sqrtm
import itertools
from collections import defaultdict
#####################################
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
        
    def calculate_expressivity(self, circuit_func, param_samples=1000, bins=50):
        """
        Calculate expressivity of quantum circuit using KL divergence
        Higher expressivity means the circuit can generate more diverse quantum states
        """
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
    
    def calculate_entanglement_capability(self, circuit_func, param_samples=100):
        """
        Calculate the entanglement capability using Meyer-Wallach measure
        """
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
        """Calculate partia l trace over specified qubit"""
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
    
    def calculate_gradient_variance(self, circuit_func, cost_func, param_samples=50):
        """
        Calculate gradient variance to detect barren plateaus
        High variance indicates good trainability, low variance indicates barren plateaus
        """
        param_ranges = self._get_parameter_ranges(circuit_func)
        gradients = []
        
        for _ in range(param_samples):
            random_params = self._generate_random_params(param_ranges)
            try:
                # Calculate numerical gradient
                eps = 1e-4
                grad = []
                
                for i in range(len(random_params)):
                    params_plus = random_params.copy()
                    params_minus = random_params.copy()
                    params_plus[i] += eps
                    params_minus[i] -= eps
                    
                    cost_plus = cost_func(params_plus)
                    cost_minus = cost_func(params_minus)
                    
                    grad_i = (cost_plus - cost_minus) / (2 * eps)
                    grad.append(grad_i)
                
                gradients.append(grad)
            except:
                continue
        
        if len(gradients) == 0:
            return 0.0
        
        gradients = np.array(gradients)
        gradient_variance = np.var(gradients, axis=0)
        
        return np.mean(gradient_variance)
    
    def calculate_quantum_volume(self, circuit_func, trials=50):
        """
        Calculate a simplified quantum volume-like metric
        Measures the circuit's ability to implement random unitaries
        """
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
    
    def calculate_circuit_fidelity(self, circuit1_func, circuit2_func, param_samples=50):
        """
        Calculate average fidelity between two quantum circuits
        Useful for comparing different circuit architectures
        """
        @qml.qnode(self.dev)
        def circuit1(params):
            circuit1_func(params)
            return qml.state()
        
        @qml.qnode(self.dev)
        def circuit2(params):
            circuit2_func(params)
            return qml.state()
        
        param_ranges = self._get_parameter_ranges(circuit1_func)
        fidelities = []
        
        for _ in range(param_samples):
            random_params = self._generate_random_params(param_ranges)
            
            try:
                state1 = circuit1(random_params)
                state2 = circuit2(random_params)
                
                if hasattr(state1, 'numpy'):
                    state1 = state1.numpy()
                if hasattr(state2, 'numpy'):
                    state2 = state2.numpy()
                
                # Calculate fidelity
                fidelity = abs(np.vdot(state1, state2))**2
                fidelities.append(fidelity)
            except:
                continue
        
        if len(fidelities) == 0:
            return 0.0
        
        return np.mean(fidelities)
    
    def _get_parameter_ranges(self, circuit_func):
        """Get parameter ranges for random sampling"""
        # Default parameter ranges for quantum circuits
        return [(-np.pi, np.pi)] * self._count_parameters(circuit_func)
    
    def _count_parameters(self, circuit_func):
        """Estimate number of parameters in circuit"""
        # This is a simplified estimation
        return self.n_qubits * 6  # Assuming 6 parameters per qubit on average
    
    def _generate_random_params(self, param_ranges):
        """Generate random parameters within specified ranges"""
        params = []
        for min_val, max_val in param_ranges:
            params.append(np.random.uniform(min_val, max_val))
        return np.array(params)
    
    def analyze_quantum_circuit(self, circuit_func, verbose=True):
        """
        Comprehensive analysis of quantum circuit
        """
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
        
        results['gradient_variance'] = self.calculate_gradient_variance(circuit_func, simple_cost)
        
        # Barren plateau indicator (low gradient variance indicates potential barren plateau)
        results['barren_plateau_risk'] = 1.0 / (1.0 + results['gradient_variance'])
        
        return results

class BMERAQuantumLayer:
    """
    Binary Matrix Ensemble of Random Assignments (BMERA) Quantum Layer
    Enhanced with quantum metrics analysis
    """
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
                qml.RZ(entangling_params[i], wires=i + 1)
            
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
                    qml.RZ(entangling_params[i], wires=i + 1)
            
            for i in range(self.n_qubits):
                qml.RX(layer2_params[i, 0], wires=i)
                #qml.RY(layer2_params[i, 1], wires=i)
                #qml.RZ(layer2_params[i, 2], wires=i)
            
            for i in range(0, self.n_qubits - 1, 2):
                if i + 1 < self.n_qubits:
                    qml.CNOT(wires=[i, i + 1])
        
        # Perform comprehensive analysis
        self.circuit_analysis = self.quantum_metrics.analyze_quantum_circuit(
            bmera_analysis_circuit, verbose=True
        )
        
        return self.circuit_analysis

class FeatureExtractor(nn.Module):
    """Classical CNN for feature extraction"""
    def __init__(self, input_channels=1, output_features=110):
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
        
        #self.flattened_size = 128 * 1 * 1
        #self.feature_map = nn.Linear(128*1*1, output_features)
        #self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(x)

        
        return x

class BMERAHybridModel(nn.Module):
    """
    Complete BMERA Hybrid Quantum-Classical Model with Quantum Metrics
    """
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
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.4),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.3),
            # nn.Linear(128, num_classes)
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


class MetricsCalculator:
    """Enhanced metrics calculator including quantum metrics"""
    
    def __init__(self, num_classes=10, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.epoch_times = []
        self.quantum_metrics_history = []
    
    def update(self, predictions, targets, probabilities=None, loss=None, quantum_metrics=None):
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())
        if loss is not None:
            self.epoch_losses.append(loss)
        if quantum_metrics is not None:
            self.quantum_metrics_history.append(quantum_metrics)
    
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
    
    def get_comprehensive_metrics(self, quantum_analysis=None):
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
            'average_loss': np.mean(self.epoch_losses) if self.epoch_losses else 0.0
        }
        
        # Add quantum metrics if available
        if quantum_analysis:
            metrics['quantum_metrics'] = quantum_analysis
        
        return metrics

def plot_quantum_metrics(quantum_analysis, save_path='quantum_metrics.png'):
    """Plot quantum circuit analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = ['expressivity', 'entanglement_capability', 'quantum_volume', 
               'gradient_variance', 'barren_plateau_risk']
    
    layer1_values = [quantum_analysis['bmera_layer1'][metric] for metric in metrics]
    layer2_values = [quantum_analysis['bmera_layer2'][metric] for metric in metrics]
    
    # Bar plot for each metric
    for i, metric in enumerate(metrics):
        axes[i].bar(['Layer 1', 'Layer 2'], [layer1_values[i], layer2_values[i]], 
                   color=['skyblue', 'lightcoral'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
    
    # Summary plot
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[5].bar(x - width/2, layer1_values, width, label='Layer 1', color='skyblue')
    axes[5].bar(x + width/2, layer2_values, width, label='Layer 2', color='lightcoral')
    axes[5].set_xlabel('Quantum Metrics')
    axes[5].set_ylabel('Score')
    axes[5].set_title('Quantum Metrics Comparison')
    axes[5].set_xticks(x)
    axes[5].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
    axes[5].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_quantum_analysis(quantum_analysis):
    """Print comprehensive quantum analysis results"""
    print("\n" + "="*80)
    print("QUANTUM CIRCUIT ANALYSIS")
    print("="*80)
    
    print("\nBMERA LAYER 1 ANALYSIS:")
    print("-" * 40)
    for metric, value in quantum_analysis['bmera_layer1'].items():
        print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
    
    print("\nBMERA LAYER 2 ANALYSIS:")
    print("-" * 40)
    for metric, value in quantum_analysis['bmera_layer2'].items():
        print(f"{metric.replace('_', ' ').title():<25}: {value:.4f}")
    
    print("\nAVERAGE QUANTUM METRICS:")
    print("-" * 40)
    print(f"{'Average Expressivity':<25}: {quantum_analysis['average_expressivity']:.4f}")
    print(f"{'Average Entanglement':<25}: {quantum_analysis['average_entanglement']:.4f}")
    print(f"{'Average Quantum Volume':<25}: {quantum_analysis['average_quantum_volume']:.4f}")
    print(f"{'Average Gradient Variance':<25}: {quantum_analysis['average_gradient_variance']:.4f}")
    print(f"{'Barren Plateau Risk':<25}: {quantum_analysis['barren_plateau_risk']:.4f}")
    
    print("\nQUANTUM METRICS INTERPRETATION:")
    print("-" * 40)
    
    # Expressivity interpretation
    avg_expr = quantum_analysis['average_expressivity']
    if avg_expr > 0.7:
        expr_status = "HIGH - Excellent state space coverage"
    elif avg_expr > 0.5:
        expr_status = "MEDIUM - Good state space coverage"
    else:
        expr_status = "LOW - Limited state space coverage"
    print(f"Expressivity: {expr_status}")
    
    # Entanglement interpretation
    avg_ent = quantum_analysis['average_entanglement']
    if avg_ent > 0.6:
        ent_status = "HIGH - Strong entanglement generation"
    elif avg_ent > 0.3:
        ent_status = "MEDIUM - Moderate entanglement generation"
    else:
        ent_status = "LOW - Weak entanglement generation"
    print(f"Entanglement: {ent_status}")
    
    # Barren plateau risk interpretation
    bp_risk = quantum_analysis['barren_plateau_risk']
    if bp_risk > 0.7:
        bp_status = "HIGH - Potential training difficulties"
    elif bp_risk > 0.4:
        bp_status = "MEDIUM - Monitor training progress"
    else:
        bp_status = "LOW - Good trainability expected"
    print(f"Barren Plateau Risk: {bp_status}")
    
    print("="*80)

def plot_confusion_matrix(conf_matrix, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_to_json(metrics, filepath='metrics_results.json'):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def print_publication_metrics(metrics):
    """Print metrics in publication format"""
    print("\n" + "="*80)
    print("PUBLICATION-QUALITY METRICS")
    print("="*80)
    
    print(f"Top-1 Accuracy: {metrics['accuracy']['top_1']:.2f}%")
    print(f"Top-5 Accuracy: {metrics['accuracy']['top_5']:.2f}%")
    print(f"Average Loss: {metrics['average_loss']:.4f}")
    
    print("\nMACRO-AVERAGED METRICS:")
    print(f"  Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_avg']['f1_score']:.4f}")
    
    print("\nMICRO-AVERAGED METRICS:")
    print(f"  Precision: {metrics['micro_avg']['precision']:.4f}")
    print(f"  Recall: {metrics['micro_avg']['recall']:.4f}")
    print(f"  F1-Score: {metrics['micro_avg']['f1_score']:.4f}")
    
    print("\nWEIGHTED-AVERAGED METRICS:")
    print(f"  Precision: {metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall: {metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1-Score: {metrics['weighted_avg']['f1_score']:.4f}")
    
    print("\nPER-CLASS METRICS:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for class_name, metrics_dict in metrics['per_class'].items():
        print(f"{class_name:<12} {metrics_dict['precision']:<10.4f} "
              f"{metrics_dict['recall']:<10.4f} {metrics_dict['f1_score']:<10.4f} "
              f"{metrics_dict['support']:<10}")
    
    # Print quantum metrics if available
    if 'quantum_metrics' in metrics:
        print_quantum_analysis(metrics['quantum_metrics'])
    
    print("="*80)
    
    
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset
import numpy as np

class FashionMnistSubset(Dataset):
    def __init__(self, root, train, download, transform, class_indices):
        self.dataset = FashionMNIST(root=root, train=train, download=download, transform=transform)
        self.class_indices = class_indices  # e.g., list(range(10, 20)) for classes 10–19

        # Filter data
        mask = [label in class_indices for label in self.dataset.targets]
        self.data = np.array(self.dataset.data)[mask]
        self.targets = np.array(self.dataset.targets)[mask]

        # Remap labels to 0–9
        class_map = {orig: new for new, orig in enumerate(class_indices)}
        self.targets = np.array([class_map[t] for t in self.targets])

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = self.dataset.transform(img)
        return img, label

    def __len__(self):
        return len(self.targets)



# Data preparation
def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = FashionMNIST(root='./dataFashion', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root='./dataFashion', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader



#class_indices = list(range(0, 10))
train_loader, test_loader = get_data_loaders(batch_size=32)

model = BMERAHybridModel(num_classes=10)  # Still 10 classes

def evaluate_model(model, test_loader, criterion, class_names):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=len(class_names), class_names=class_names)
    
    total_time = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            start_time = time.time()
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            metrics_calc.update(predicted, target, probabilities, loss.item())
            
            total_time += time.time() - start_time
    
    # Get quantum analysis if available
    quantum_analysis = getattr(model, 'quantum_analysis_results', None)
    
    final_metrics = metrics_calc.get_comprehensive_metrics(quantum_analysis)
    final_metrics['inference_time'] = total_time / len(test_loader.dataset)
    
    return final_metrics



def train_model(model, train_loader, test_loader, epochs=50, lr=0.001):
    """Enhanced training function with quantum metrics analysis"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
    #                'dog', 'frog', 'horse', 'ship', 'truck']
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Analyze quantum circuit properties before training
    print("Analyzing quantum circuit properties...")
    quantum_analysis = model.analyze_quantum_components()
    
    # Print quantum analysis
    print_quantum_analysis(quantum_analysis)
    
    # Plot quantum metrics
    plot_quantum_metrics(quantum_analysis, 'quantum_analysis.png')
    
    best_accuracy = 0.0
    best_metrics = None
    training_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_metrics = MetricsCalculator(num_classes=10, class_names=class_names)
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update training metrics
            _, predicted = torch.max(output, 1)
            train_metrics.update(predicted, target, loss=loss.item())
            
            # Update progress bar
            current_acc = train_metrics.calculate_accuracy()
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        # Testing phase with comprehensive evaluation
        test_metrics = evaluate_model(model, test_loader, criterion, class_names)
        
        # Record training history
        train_acc = train_metrics.calculate_accuracy()
        training_history['train_loss'].append(np.mean(train_metrics.epoch_losses))
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_metrics['average_loss'])
        training_history['test_acc'].append(test_metrics['accuracy']['top_1'])
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nEpoch {epoch+1}/{epochs} (Time: {epoch_time:.2f}s):')
        print(f'Train Loss: {training_history["train_loss"][-1]:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_metrics["average_loss"]:.4f}, Test Acc: {test_metrics["accuracy"]["top_1"]:.2f}%')
        print(f'Test Top-5 Acc: {test_metrics["accuracy"]["top_5"]:.2f}%')
        print(f'Test F1 (Macro): {test_metrics["macro_avg"]["f1_score"]:.4f}')
        print('-' * 80)
        
        # Save best model
        if test_metrics['accuracy']['top_1'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']['top_1']
            best_metrics = test_metrics
            torch.save(model.state_dict(), 'bmera_best_model.pth')
            print(f'New best model saved with Top-1 accuracy: {best_accuracy:.2f}%')
        
        scheduler.step()
    
    # Final comprehensive evaluation with quantum metrics
    print("\nFINAL MODEL EVALUATION:")
    print_publication_metrics(best_metrics)
    
    # Save results
    save_metrics_to_json(best_metrics, 'final_metrics_with_quantum.json')
    plot_confusion_matrix(np.array(best_metrics['confusion_matrix']), class_names)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save quantum analysis separately
    with open('quantum_analysis.json', 'w') as f:
        json.dump(quantum_analysis, f, indent=2)
    
    return best_accuracy, best_metrics

def analyze_quantum_performance_correlation(training_history, quantum_analysis):
    """Analyze correlation between quantum metrics and model performance"""
    print("\n" + "="*80)
    print("QUANTUM-CLASSICAL PERFORMANCE CORRELATION ANALYSIS")
    print("="*80)
    
    final_accuracy = training_history['test_acc'][-1]
    convergence_rate = (training_history['test_acc'][-1] - training_history['test_acc'][0]) / len(training_history['test_acc'])
    
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print(f"Convergence Rate: {convergence_rate:.4f}% per epoch")
    
    # Correlate with quantum metrics
    avg_expressivity = quantum_analysis['average_expressivity']
    avg_entanglement = quantum_analysis['average_entanglement']
    barren_plateau_risk = quantum_analysis['barren_plateau_risk']
    
    print(f"\nQuantum Metrics vs Performance:")
    print(f"Expressivity (0-1): {avg_expressivity:.4f}")
    print(f"Entanglement (0-1): {avg_entanglement:.4f}")
    print(f"Barren Plateau Risk (0-1): {barren_plateau_risk:.4f}")
    
    # Simple correlation analysis
    print(f"\nPerformance Indicators:")
    if avg_expressivity > 0.6 and final_accuracy > 80:
        print("✓ High expressivity correlates with good performance")
    elif avg_expressivity < 0.4 and final_accuracy < 70:
        print("⚠ Low expressivity may limit performance")
    
    if avg_entanglement > 0.5 and convergence_rate > 0.5:
        print("✓ Good entanglement capability supports learning")
    elif avg_entanglement < 0.3 and convergence_rate < 0.3:
        print("⚠ Low entanglement may slow convergence")
    
    if barren_plateau_risk > 0.7 and convergence_rate < 0.2:
        print("⚠ High barren plateau risk detected - consider architecture changes")
    elif barren_plateau_risk < 0.4:
        print("✓ Low barren plateau risk - good trainability")
    
    print("="*80)

def generate_quantum_report(model, quantum_analysis, final_metrics, save_path='quantum_report.txt'):
    """Generate comprehensive quantum circuit analysis report"""
    
    with open(save_path, 'w') as f:
        f.write("BMERA QUANTUM-CLASSICAL HYBRID MODEL ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Model architecture summary
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of Qubits: {model.n_qubits}\n")
        f.write(f"Number of Classes: {model.num_classes}\n")
        f.write(f"Quantum Layers: 2 (BMERA Layer 1 & 2)\n")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        quantum_params = sum(p.numel() for name, p in model.named_parameters() 
                           if 'bmera_layer' in name and p.requires_grad)
        classical_params = total_params - quantum_params
        
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Quantum Parameters: {quantum_params:,} ({100*quantum_params/total_params:.1f}%)\n")
        f.write(f"Classical Parameters: {classical_params:,} ({100*classical_params/total_params:.1f}%)\n\n")
        
        # Quantum metrics analysis
        f.write("QUANTUM CIRCUIT ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        for layer_name, layer_analysis in [('BMERA Layer 1', quantum_analysis['bmera_layer1']),
                                          ('BMERA Layer 2', quantum_analysis['bmera_layer2'])]:
            f.write(f"\n{layer_name}:\n")
            for metric, value in layer_analysis.items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value:.4f}\n")
        
        f.write(f"\nOverall Quantum Performance:\n")
        f.write(f"  Average Expressivity: {quantum_analysis['average_expressivity']:.4f}\n")
        f.write(f"  Average Entanglement: {quantum_analysis['average_entanglement']:.4f}\n")
        f.write(f"  Average Quantum Volume: {quantum_analysis['average_quantum_volume']:.4f}\n")
        f.write(f"  Barren Plateau Risk: {quantum_analysis['barren_plateau_risk']:.4f}\n")
        
        # Classification performance
        f.write(f"\nCLASSIFICATION PERFORMANCE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Top-1 Accuracy: {final_metrics['accuracy']['top_1']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {final_metrics['accuracy']['top_5']:.2f}%\n")
        f.write(f"Macro F1-Score: {final_metrics['macro_avg']['f1_score']:.4f}\n")
        f.write(f"Weighted F1-Score: {final_metrics['weighted_avg']['f1_score']:.4f}\n")
        
        # Recommendations
        f.write(f"\nRECOMMENDations:\n")
        f.write("-" * 20 + "\n")
        
        if quantum_analysis['average_expressivity'] > 0.7:
            f.write("✓ Excellent expressivity - circuit can explore diverse quantum states\n")
        elif quantum_analysis['average_expressivity'] < 0.4:
            f.write("⚠ Consider increasing circuit depth or adding more variational layers\n")
        
        if quantum_analysis['average_entanglement'] > 0.6:
            f.write("✓ Strong entanglement capability - good for quantum advantage\n")
        elif quantum_analysis['average_entanglement'] < 0.3:
            f.write("⚠ Consider adding more entangling gates or different connectivity\n")
        
        if quantum_analysis['barren_plateau_risk'] > 0.7:
            f.write("⚠ High barren plateau risk - consider parameter initialization strategies\n")
        else:
            f.write("✓ Low barren plateau risk - good trainability expected\n")
    
    print(f"Comprehensive quantum report saved to: {save_path}")

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 16
    N_QUBITS = 6
    NUM_CLASSES = 10
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    print("="*80)
    print("BMERA QUANTUM-CLASSICAL HYBRID MODEL WITH QUANTUM METRICS")
    print("="*80)
    
    # Get data loaders
    print("Loading Fashion_mnist dataset...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Create model
    print("Creating BMERA Hybrid Quantum-Classical Model...")
    model = BMERAHybridModel(n_qubits=N_QUBITS, num_classes=NUM_CLASSES).to(device)
    
    # Count and display parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(p.numel() for name, p in model.named_parameters() 
                        if 'bmera_layer' in name and p.requires_grad)
    classical_params = total_params - quantum_params
    
    print(f"Model Configuration:")
    print(f"  Total trainable parameters: {total_params:,}")
    print(f"  Quantum parameters: {quantum_params:,} ({100*quantum_params/total_params:.1f}%)")
    print(f"  Classical parameters: {classical_params:,} ({100*classical_params/total_params:.1f}%)")
    
    # Train model with quantum analysis
    print("\nStarting training with quantum circuit analysis...")
    best_acc, final_metrics = train_model(model, train_loader, test_loader, 
                                         epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Load training history for correlation analysis
    with open('training_history.json', 'r') as f:
        training_history = json.load(f)
    
    # Perform quantum-classical correlation analysis
    analyze_quantum_performance_correlation(training_history, model.quantum_analysis_results)
    
    # Generate comprehensive report
    generate_quantum_report(model, model.quantum_analysis_results, final_metrics)
    
    print(f"\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Best test accuracy achieved: {best_acc:.2f}%")
    print("\nGenerated files:")
    print("- final_metrics_with_quantum.json (Complete metrics)")
    print("- quantum_analysis.json (Quantum circuit analysis)")
    print("- quantum_analysis.png (Quantum metrics visualization)")
    print("- confusion_matrix.png (Classification results)")
    print("- training_history.json (Training progress)")
    print("- quantum_report.txt (Comprehensive analysis report)")
    print("- bmera_best_model.pth (Best model weights)")
    print("="*80) 
