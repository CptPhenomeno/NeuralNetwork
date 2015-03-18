namespace NeuralNetwork.Activation
{
    public interface IActivationFunction
    {
        double Function (double x);
        double Derivative(double x);
    }
}
