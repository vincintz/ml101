# ML101 - Neural Network Implementation

A Java implementation of a multilayer perceptron neural network for learning purposes. The project demonstrates basic neural network concepts without external dependencies.

## Requirements

- Java 25 (OpenJDK 25 recommended)
- Gradle (uses wrapper, so no local installation needed)

## Project Structure

- `mlp/` - Core neural network implementation
  - Includes XOR and MNIST examples
- `data/mnist/` - MNIST dataset location (needs to be downloaded)

## Building and Testing

Basic build:
```bash
./gradlew build
```

By default, MNIST-related tests will be skipped if the dataset is not present. To run MNIST tests:

1. Download the MNIST dataset:
   ```bash
   # From repository root
   cd data/mnist
   chmod +x get-data.sh
   ./get-data.sh
   ```

2. Run tests again:
   ```bash
   ./gradlew test
   ```

## Examples

The project includes two example implementations:

1. XOR Problem (`mlp/src/test/java/ml101/mlp/XorTest.java`)
   - Demonstrates basic network training on the XOR logical operation
   - Runs without additional dependencies

2. MNIST Digit Recognition (`mlp/src/test/java/ml101/mlp/MNISTTest.java`)
   - Handwritten digit classification
   - Requires downloading the MNIST dataset as described above
   - Uses a 784-1200-10 network architecture

## Implementation Details

This is a pure Java implementation focusing on clarity and educational value. Key features:

- No external ML/math dependencies
- Clear implementation of backpropagation
- Configurable activation functions
- Support for saving/loading trained networks