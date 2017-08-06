package ml101.mlp.activation;

public class StepFn implements ActivationFn {
    @Override
    public double compute(double z) {
        if (z < 0)
            return 0.0;
        else
            return 1.0;
    }
}
