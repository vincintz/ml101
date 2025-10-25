package ml101.mlp;

import static org.junit.Assert.assertEquals;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import ml101.mlp.activation.LogisticFn;
import ml101.mlp.data.MNISTData;

public class MNISTTest {
    private MNISTData trainingData;
    private MNISTData testData;

    @Before
    public void loadData() {
    // If MNIST data isn't present in ../data/mnist, skip these tests.
    boolean hasData = Files.exists(Paths.get("../data/mnist/train-images-idx3-ubyte"))
        && Files.exists(Paths.get("../data/mnist/train-labels-idx1-ubyte"))
        && Files.exists(Paths.get("../data/mnist/t10k-images-idx3-ubyte"))
        && Files.exists(Paths.get("../data/mnist/t10k-labels-idx1-ubyte"));
    Assume.assumeTrue("MNIST data not found in ../data/mnist — skipping MNIST tests", hasData);

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
                .activation(new LogisticFn(2.0))
                .stopWhen((iteration, cost) -> iteration >= 1000 || cost < 0.1)
                .reportStatus((iteration, cost) -> System.out.println(iteration + "\t" + cost))
                .load();
        mlp.train(trainingData);
        mlp.save("mnist1200.mlp");
    }

}