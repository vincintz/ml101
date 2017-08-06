package ml101.mlp;

import ml101.mlp.ml101.mlp.activation.StepFn;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class MLPTest {
    @Test
    public void shouldWorkWithManuallyConfiguredXOR() {
        final double DELTA = 0.001;
        final MLP mlp =
                new MLP.Config()
                        .activation(new StepFn())
                        .layers(2, 2, 1)
                        .weights( 0.5,  1.0,  1.0,
                                 -1.5, -1.0, -1.0,
                                  1.5,  1.0,  1.0)
                        .build();
        assertEquals(mlp.compute(0.0, 0.0), 0.0, DELTA);
        assertEquals(mlp.compute(0.0, 1.0), 1.0, DELTA);
        assertEquals(mlp.compute(1.0, 0.0), 1.0, DELTA);
        assertEquals(mlp.compute(1.0, 1.0), 0.0, DELTA);
    }

    @Test
    public void sampleMatrixOperations() {
        final INDArray[] data = new INDArray[] {
                Nd4j.create(new double[] {-1.0, 0.0, 0.0}, new int[] {3}),
                Nd4j.create(new double[] {-1.0, 0.0, 1.0}, new int[] {3}),
                Nd4j.create(new double[] {-1.0, 1.0, 0.0}, new int[] {3}),
                Nd4j.create(new double[] {-1.0, 1.0, 1.0}, new int[] {3})
        };
        final INDArray weights = Nd4j.create(
                new double[] { 0.5,  1.0,  1.0, -1.5, -1.0, -1.0},
                new int[] {2, 3});
        System.out.println(weights);
        for (final INDArray d: data) {
            System.out.println(weights.mmul(d.transpose()));
        }
    }
}