package ml101.mlp.activation;

import java.io.Serializable;

public class LogisticFn implements ActivationFn, Serializable {
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
