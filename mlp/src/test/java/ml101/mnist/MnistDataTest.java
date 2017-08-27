package ml101.mnist;

import mnist.MnistData;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MnistDataTest {
    @Test
    public void testReadData() {
        final MnistData trainingData = new MnistData(
                "../data/mnist/train-images-idx3-ubyte",
                "../data/mnist/train-labels-idx1-ubyte");
        final MnistData testData = new MnistData(
                "../data/mnist/t10k-images-idx3-ubyte",
                "../data/mnist/t10k-labels-idx1-ubyte");
        assertEquals(60000, trainingData.numberOfItems());
        assertEquals(10000, testData.numberOfItems());
    }

}