package ml101.mlp.activation;

import java.io.Serializable;

public class StepFn implements ActivationFn, Serializable {
    @Override
    public double compute(double z) {
        if (z < 0)
            return 0.0;
        else
            return 1.0;
    }

    @Override
    public double derivative(double z) {
        throw new ArithmeticException("StepFn does not support derivative");
    }
}
