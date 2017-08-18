package ml101.mlp;

import ml101.mlp.activation.LogisticFn;
import ml101.mlp.activation.StepFn;
import org.junit.Test;

import static org.junit.Assert.*;

public class MLPTest {
    private final double DELTA = 0.001;

    @Test
    public void shouldWorkWithManuallyConfiguredXOR() {
        final MLP mlp = new MLP.Builder()
                .activation(new StepFn())
                .layers(2, 2, 1)
                .weights( -0.5,  1.0,  1.0,
                           1.5, -1.0, -1.0,
                          -1.5,  1.0,  1.0)
                .build();
        assertEquals(0.0, mlp.compute(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.compute(1.0, 1.0)[0], DELTA);
    }

    @Test
    public void shouldWorkWithTrainedConfiguredXOR() {
        final MLP mlp = new MLP.Builder()
                .activation(new LogisticFn())
                .layers(2, 2, 1)
                .randomWeights()
                .learningRate(0.05)
                .epochs(100)
                .build();
        mlp.train(
                new double[][] {{0.0, 0.0},
                                {0.0, 1.0},
                                {1.0, 0.0},
                                {1.0, 1.0}},
                new double[][] {{0.0},
                                {1.0},
                                {1.0},
                                {0.0}});
        assertEquals(0.0, mlp.compute(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.compute(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.compute(1.0, 1.0)[0], DELTA);
    }

}