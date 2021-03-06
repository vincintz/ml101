package ml101.mlp;

import ml101.mlp.activation.ActivationFn;

import java.io.*;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;

import ml101.mlp.data.TrainingData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Multi-layer Perceptron
 */
public class MLP implements Serializable {
    transient private final static Logger logger = LoggerFactory.getLogger(MLP.class);
    private double learningRate;
    transient private BiPredicate<Long, Double> stopCriteria;
    transient private BiConsumer<Long, Double> reporter;

    final Layer[] layers;

    /**
     * MLP Constructor
     */
    private MLP(final ActivationFn activationFn, final int[] nodesPerLayer) {
        layers = new Layer[nodesPerLayer.length - 1];
        for (int l = 0; l < layers.length; l++) {
            layers[l] = new Layer(activationFn, nodesPerLayer[l], nodesPerLayer[l + 1]);
        }
    }

    /*
     * Demo method for manually setting weights and biases. Useful for testing manually selecting weights and biases
     * for an XOR mlp.
     */
    private void setWeightsAndBiases(double[] rawWeights) {
        int start = 0;
        for (int l = 0; l < layers.length; l++) {
            start = layers[l].setWeightsAndBiases(start, rawWeights);
        }
    }

    /*
     * Initializes computation buffers. These arrays are used to avoid multiple calls to 'new'.
     */
    private void initializeComputationBuffers() {
        for (int l = 0; l < layers.length; l++) {
            layers[l].initializeComputationBuffers();
        }
    }


    /**
     * Feed forward computation
     */
    public double[] feedForward(double... input) {
        double[] layerOutput = input;
        for (int l = 0; l < layers.length; l++) {
            layerOutput = layers[l].feedForward(layerOutput);
        }
        return layerOutput;
    }

    /**
     * Train using batch back propagation
     */
    public void train(final TrainingData trainingData) {
        long iteration = 0;
        double totalCost;
        do {
            totalCost = 0.0;
            for (int n = 0; n < trainingData.length(); n++) {
                double[] output = feedForward(trainingData.input(n));
                totalCost += computeCost(trainingData.output(n), output);
                computeDeltaWeightsAndBias(trainingData.output(n), output, trainingData.input(n));
            }
            updateTotalWeightsAndBias();
            // use reportStatus lambda (BiConsumer) to display current status
            reporter.accept(++iteration, totalCost / trainingData.length());
        } while (!stopCriteria.test(iteration, totalCost));
    }

    /**
     * @return Returns the 'cost' at the output layers
     */
    private double computeCost(double[] expected, double[] output) {
        double cost = 0;
        for (int j = 0; j < expected.length; j++) {
            cost += -expected[j] * Math.log(output[j])
                    - (1.0-expected[j]) * Math.log(1.0 - output[j]);
        }
        return cost;
    }

    /*
     * Used to generate gradient graph
     */
    double computeCost(final TrainingData trainingData) {
        double cost = 0.0;
        for (int n = 0; n < trainingData.length(); n++) {
            double[] output = feedForward(trainingData.input(n));
            cost += computeCost(trainingData.output(n), output);
        }
        return cost / trainingData.length();
    }
    /**
     * Computes the change in weights and biases, starting at the output layers, going backwards.
     */
    private void computeDeltaWeightsAndBias(double[] expected, double[] output, double[] input) {
        layers[layers.length-1].computeErrorAtOutputLayer(expected, output);
        for (int l = layers.length-1; l > 0; l--) {
            final Layer currentLayer = layers[l];
            final Layer previousLayer = layers[l-1];
            previousLayer.propagateErrors(currentLayer);
            currentLayer.computeDeltaWeightsAndBias(previousLayer.output, learningRate);
        }
        layers[0].computeDeltaWeightsAndBias(input, learningRate);
    }

    /**
     * Updates the network weigts and biases
     */
    private void updateTotalWeightsAndBias() {
        for (int l = 0; l < layers.length; l++) {
            layers[l].updateTotalWeightsAndBias();
        }
    }

    /**
     * Helper function for displaying the weights
     * @param text
     */
    public void displayWeightsAndBias(final String text) {
        logger.info(text);
        for (int l = 0; l < layers.length; l++) {
            logger.info("Layer " + (l+1));
            layers[l].displayWeightsAndBias();
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
        transient private BiPredicate<Long, Double> stopCriteria = (iteration, cost) -> iteration < 1000;
        transient private BiConsumer<Long, Double> reporter = (iteration, cost) -> {
            if (iteration % 1000 == 0) {
                logger.info("\t{}\t{}", iteration, cost);
            }
        };

        public MLP load() {
            final MLP mlp;
            mlp = new MLP(activationFn, nodesPerLayer);
            mlp.learningRate = learningRate;
            mlp.reporter = reporter;
            mlp.stopCriteria = stopCriteria;
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

        public Builder reportStatus(BiConsumer<Long, Double> reporter) {
            this.reporter = reporter;
            return this;
        }

        public Builder stopWhen(BiPredicate<Long, Double> stopCriteria) {
            this.stopCriteria = stopCriteria;
            return this;
        }

        public MLP load(String filename) throws Exception {
            try (ObjectInputStream stream
                         = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filename)))) {
                final MLP mlp = (MLP)stream.readObject();
                mlp.initializeComputationBuffers();
                return mlp;
            }
        }
    }

}
