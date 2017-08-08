package ml101.mlp.activation;

public class SigmoidFn implements ActivationFn {
    @Override
    public double compute(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double derivative(double z) {
        double fz = compute(z);
        return fz * (1.0 - fz);
    }
}
