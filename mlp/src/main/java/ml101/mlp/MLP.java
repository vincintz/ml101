package ml101.mlp;

import ml101.mlp.activation.ActivationFn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Multi Layered Perceptron
 */
public class MLP {
    private final ActivationFn activationFn;
    private final INDArray[] weights;

    private MLP(final ActivationFn activationFn, final INDArray... weights) {
        this.activationFn = activationFn;
        this.weights = weights;
    }

    public double compute(double... x) {
        return Double.NaN;
    }

    /**
     * MLP configuration object.
     * Collects configuration info, then builds a MLP.
     */
    public static class Config {
        private ActivationFn activationFn;
        private int[] nodesPerLayer;
        private double[] rawWeights;

        public MLP build() {
            final int numLayers = nodesPerLayer.length - 1;
            INDArray[] weights = new INDArray[numLayers];
            int w = 0;
            for (int l = 0; l < nodesPerLayer.length - 1; l++) {
                weights[l] = Nd4j.zeros(nodesPerLayer);
            }
            final MLP mlp = new MLP(activationFn, weights);
            return mlp;
        }

        public Config activation(final ActivationFn fn) {
            this.activationFn = fn;
            return this;
        }

        public MLP.Config layers(final int... ll) {
            this.nodesPerLayer = ll;
            return this;
        }

        public Config weights(double... ww) {
            this.rawWeights = ww;
            return this;
        }

    }
}
