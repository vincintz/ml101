package ml101.mlp;

import ml101.mlp.activation.LogisticFn;
import ml101.mlp.data.MNISTData;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

public class MNISTTest {
    transient private final static Logger logger = LoggerFactory.getLogger(MNISTTest.class);
    private MNISTData trainingData;
    private MNISTData testData;

    @Before
    public void loadData() {
        trainingData = new MNISTData(
                "../data/mnist/train-images-idx3-ubyte",
                "../data/mnist/train-labels-idx1-ubyte");
        testData = new MNISTData(
                "../data/mnist/t10k-images-idx3-ubyte",
                "../data/mnist/t10k-labels-idx1-ubyte");
    }

    @Test
    public void shouldLoadData() {
        assertEquals(60000, trainingData.length());
        assertEquals(10000, testData.length());
    }

    @Test
    public void shouldTrainAndSave() throws Exception {
        final MLP mlp = new MLP.Builder()
                .layers(28*28, 1200, 10)
                .activation(new LogisticFn())
                .iterations(500000)
                .reporter((epoch, cost) -> {
                    logger.info("\t{}\t{}", epoch, cost);
                })
                .build();
        mlp.train(trainingData);
        mlp.save("mnist1200.mlp");
    }

}