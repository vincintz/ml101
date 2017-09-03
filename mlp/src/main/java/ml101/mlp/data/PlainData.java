package ml101.mlp.data;

import static java.lang.Integer.min;

public class PlainData implements TrainingData {
    final private int length;
    final private double[][] input;
    final private double[][] output;

    public PlainData(double[][] input, double output[][]) {
        this.length = min(input.length, output.length);
        this.input = input;
        this.output = output;
    }

    @Override
    public int length() {
        return this.length;
    }

    @Override
    public double[] input(int n) {
        return this.input[n];
    }

    @Override
    public double[] output(int n) {
        return this.output[n];
    }
}
