package ml101.mlp.activation;

public interface ActivationFn {
    double compute(double z);
    double derivative(double z);
}
