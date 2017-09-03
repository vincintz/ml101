package ml101.mlp;

import ml101.mlp.activation.LogisticFn;
import ml101.mlp.activation.StepFn;
import ml101.mlp.data.PlainData;
import org.junit.Test;

import static org.junit.Assert.*;

public class XorTest {
    private final double DELTA = 0.5;

    @Test
    public void shouldWorkWithManuallyConfiguredXOR() {
        final MLP mlp = new MLP.Builder()
                .activation(new LogisticFn(5.0))
                .layers(2, 2, 1)
                .weights( -0.5,  1.0,  1.0,
                           1.5, -1.0, -1.0,
                          -1.5,  1.0,  1.0)
                .build();
        mlp.displayWeightsAndBias("Manually Configured");
        assertEquals(0.0, mlp.feedForward(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.feedForward(1.0, 1.0)[0], DELTA);
    }

    @Test
    public void shouldWorkWithTrainedXOR() {
        final MLP mlp = new MLP.Builder()
                .activation(new LogisticFn(1.0))
                .layers(2, 2, 1)
                .learningRate(0.05)
                .iterations(500000)
                .build();
        mlp.displayWeightsAndBias("Before Training");
        mlp.train(new PlainData(
                new double[][] {{0.0, 0.0},
                                {0.0, 1.0},
                                {1.0, 0.0},
                                {1.0, 1.0}},
                new double[][] {{0.0},
                                {1.0},
                                {1.0},
                                {0.0}}));
        mlp.displayWeightsAndBias("After Training");
        assertEquals(0.0, mlp.feedForward(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.feedForward(1.0, 1.0)[0], DELTA);
    }

    @Test
    public void shouldSaveAndLoadXor() throws Exception {
        final MLP mlp = new MLP.Builder()
                .activation(new StepFn())
                .layers(2, 2, 1)
                .weights( -0.5,  1.0,  1.0,
                        1.5, -1.0, -1.0,
                        -1.5,  1.0,  1.0)
                .build();
        mlp.save("xor.mlp");
        final MLP xorLoaded = new MLP.Builder()
                .build("xor.mlp");
        xorLoaded.displayWeightsAndBias("Loaded from file");
        assertEquals(0.0, xorLoaded.feedForward(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, xorLoaded.feedForward(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, xorLoaded.feedForward(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, xorLoaded.feedForward(1.0, 1.0)[0], DELTA);
    }
}