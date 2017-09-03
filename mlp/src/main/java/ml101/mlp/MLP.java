package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.io.*;
import java.util.function.BiConsumer;

import ml101.mlp.data.TrainingData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Multi-layer Perceptron
 */
public class MLP implements Serializable {
    transient private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    transient private double learningRate;
    transient private int epochs;
    transient private BiConsumer<Integer, Double> reporter;

    final private Layer[] layer;

    /**
     * MLP Constructor
     */
    private MLP(final ActivationFn activationFn, final int[] nodesPerLayer) {
        final int numLayers = nodesPerLayer.length - 1;
        layer = new Layer[numLayers];
        for (int l = 0; l < numLayers; l++) {
            layer[l] = new Layer(activationFn, nodesPerLayer[l], nodesPerLayer[l + 1]);
        }
    }

    /*
     * Demo method for manually setting weights and biases. Useful for testing manually selecting weights and biases
     * for an XOR mlp.
     */
    private void setWeightsAndBiases(double[] rawWeights) {
        int start = 0;
        for (int l = 0; l < layer.length; l++) {
            start = layer[l].setWeightsAndBiases(start, rawWeights);
        }
    }

    /*
     * Initializes computation buffers. These arrays are used to avoid multiple calls to 'new'.
     */
    private void initializeComputationBuffers() {
        for (int l = 0; l < layer.length; l++) {
            layer[l].initializeComputationBuffers();
        }
    }


    /**
     * Feed forward computation
     */
    public double[] feedForward(double... input) {
        double[] layerOutput = input;
        for (int l = 0; l < layer.length; l++) {
            layerOutput = layer[l].feedForward(layerOutput);
        }
        return layerOutput;
    }

    /**
     * Train using batch back propagation
     */
    public void train(final TrainingData trainingData) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double cost = doBatchBackProp(trainingData);
            reporter.accept(epoch, cost);
        }
    }

    /**
     * One back-propagation epoch.
     */
    private double doBatchBackProp(final TrainingData trainingData) {
        double totalCost = 0.0;
        for (int n = 0; n < trainingData.length(); n++) {
            double[] output = feedForward(trainingData.input(n));
            totalCost += computeCost(trainingData.output(n), output);
            computeDeltaWeightsAndBias(trainingData.output(n), output, trainingData.input(n));
        }
        updateTotalWeightsAndBias();
        return totalCost / trainingData.length();
    }

    /**
     * @return Returns the 'cost' at the output layer
     */
    private double computeCost(double[] expected, double[] output) {
        double cost = 0;
        for (int j = 0; j < expected.length; j++) {
            cost += -expected[j] * Math.log(output[j])
                    - (1.0-expected[j]) * Math.log(1.0 - output[j]);
        }
        return cost;
    }

    /**
     * Computes the change in weights and biases, starting at the output layer, going backwards.
     */
    private void computeDeltaWeightsAndBias(double[] expected, double[] output, double[] input) {
        layer[layer.length-1].computeErrorAtOutputLayer(expected, output);
        for (int l = layer.length-1; l > 0; l--) {
            final Layer currentLayer = layer[l];
            final Layer previousLayer = layer[l-1];
            previousLayer.propagateErrors(currentLayer);
            currentLayer.computeDeltaWeightsAndBias(previousLayer.output, learningRate);
        }
        layer[0].computeDeltaWeightsAndBias(input, learningRate);
    }

    /**
     * Updates the network weigts and biases
     */
    private void updateTotalWeightsAndBias() {
        for (int l = 0; l < layer.length; l++) {
            layer[l].updateTotalWeightsAndBias();
        }
    }

    /**
     * Helper function for displaying the weights
     * @param text
     */
    public void displayWeightsAndBias(final String text) {
        logger.info(text);
        for (int l = 0; l < layer.length; l++) {
            logger.info("Layer " + (l+1));
            layer[l].displayWeightsAndBias();
        }
    }

    /**
     * Save the network in a file
     */
    public void save(String filename) throws IOException {
        final File file = new File(filename);
        try (ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            out.writeObject(this);
            out.flush();
        }
    }

    /**
     * MLP configuration object.
     * Collects configuration info, then builds a MLP.
     */
    public static class Builder {
        transient private ActivationFn activationFn = null;
        transient private int[] nodesPerLayer = null;
        transient private double[] rawWeights = null;
        transient private double learningRate = 0.01;
        transient private int epochs = 1000;
        transient private BiConsumer<Integer, Double> reporter = (epoch, cost) -> {
            if (epoch % 1000 == 0) {
                logger.info("\t{}\t{}", epoch, cost);
            }
        };

        public MLP build() {
            final MLP mlp;
            mlp = new MLP(activationFn, nodesPerLayer);
            mlp.learningRate = learningRate;
            mlp.epochs = epochs;
            mlp.reporter = reporter;
            mlp.initializeComputationBuffers();
            if (rawWeights != null) {
                mlp.setWeightsAndBiases(rawWeights);
            }
            return mlp;
        }

        public Builder activation(final ActivationFn fn) {
            this.activationFn = fn;
            return this;
        }

        public Builder layers(final int... nodesPerLayer) {
            this.nodesPerLayer = nodesPerLayer;
            return this;
        }

        public Builder weights(double... weights) {
            this.rawWeights = weights;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder iterations(int epochs) {
            this.epochs = epochs;
            return this;
        }
        public Builder reporter(BiConsumer<Integer, Double> reporter) {
            this.reporter = reporter;
            return this;
        }

        public MLP build(String filename) throws Exception {
            try (ObjectInputStream stream
                         = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filename)))) {
                final MLP mlp = (MLP)stream.readObject();
                mlp.initializeComputationBuffers();
                return mlp;
            }
        }
    }

}
