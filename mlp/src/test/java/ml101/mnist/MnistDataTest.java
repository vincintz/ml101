package ml101.mnist;

import ml101.mlp.MLP;
import ml101.mlp.activation.LogisticFn;
import mnist.MnistData;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MnistDataTest {
    private MnistData trainingData;
    private MnistData testData;

    @Before
    public void loadMnistData() {
        trainingData = new MnistData(
                "../data/mnist/train-images-idx3-ubyte",
                "../data/mnist/train-labels-idx1-ubyte");
        testData = new MnistData(
                "../data/mnist/t10k-images-idx3-ubyte",
                "../data/mnist/t10k-labels-idx1-ubyte");
    }

    @Test
    public void shouldLoadMnistData() {
        assertEquals(60000, trainingData.numberOfItems());
        assertEquals(10000, testData.numberOfItems());
    }

    @Test
    public void shouldTrainAndSave() throws Exception {
        final MLP mnistMlp = new MLP.Builder()
                .layers(28*28, 1200, 10)
                .activation(new LogisticFn())
                .iterations(500000)
                .build();
        mnistMlp.train(trainingData);
        mnistMlp.save("mnist1200.mlp");
    }

}