package ml101.mlp;

import ml101.mlp.activation.LogisticFn;
import ml101.mlp.activation.StepFn;
import ml101.mlp.data.PlainData;
import ml101.mlp.data.TrainingData;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

public class XorTest {
    private final static Logger logger = LoggerFactory.getLogger(XorTest.class);
    private final double DELTA = 0.1;

    @Test
    public void shouldWorkWithManuallyConfiguredXOR() {
        final MLP mlp = new MLP.Builder()
                .activation(new StepFn())
                .layers(2, 2, 1)
                .weights( -0.5,  1.0,  1.0,
                           1.5, -1.0, -1.0,
                          -1.5,  1.0,  1.0)
                .load();
        mlp.displayWeightsAndBias("Manually Configured");
        assertEquals(0.0, mlp.feedForward(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, mlp.feedForward(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, mlp.feedForward(1.0, 1.0)[0], DELTA);
    }

    @Test
    public void shouldWorkWithTrainedXOR() throws Exception {
        final MLP mlp = new MLP.Builder()
                .activation(new LogisticFn(1.0))
                .layers(2, 2, 1)
                .learningRate(0.1)
                .reportStatus((iteration, cost) -> {
                    if (iteration % 1000 == 0) {
                        logger.info("\t{}\t{}", iteration, cost);
                    }
                })
                .stopWhen((iteration, cost) -> iteration >= 500000 || cost < 0.005)
                .load();
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
                .load();
        mlp.save("xor.mlp");
        final MLP xorLoaded = new MLP.Builder().load("xor.mlp");
        xorLoaded.displayWeightsAndBias("Loaded from file");
        assertEquals(0.0, xorLoaded.feedForward(0.0, 0.0)[0], DELTA);
        assertEquals(1.0, xorLoaded.feedForward(0.0, 1.0)[0], DELTA);
        assertEquals(1.0, xorLoaded.feedForward(1.0, 0.0)[0], DELTA);
        assertEquals(0.0, xorLoaded.feedForward(1.0, 1.0)[0], DELTA);
    }

    // @ Test
    public void createDemoData() {
        final MLP mlp = new MLP.Builder()
                .activation(new LogisticFn(5.0))
                .layers(2, 2, 1)
                .weights( -0.5,  1.0,  1.0,
                        1.5, -1.0, -1.0,
                        -1.5,  1.0,  1.0)
                .load();
        TrainingData td = new PlainData(
                new double[][] {{0.0, 0.0},
                        {0.0, 1.0},
                        {1.0, 0.0},
                        {1.0, 1.0}},
                new double[][] {{0.0},
                        {1.0},
                        {1.0},
                        {0.0}});
        System.out.println("weight\tcost");
        for (double w = -2.0; w < 2.0; w += 0.01) {
            mlp.layers[0].weights[1][0] = w;
            double cost = mlp.computeCost(td);
            System.out.println(w + "\t" + cost);
        }
    }

}