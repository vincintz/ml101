package ml101.mlp.data;

public interface TrainingData {
    int length();
    double[] input(int n);
    double[] output(int n);
}
