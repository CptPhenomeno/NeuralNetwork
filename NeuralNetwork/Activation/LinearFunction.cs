namespace NeuralNetwork.Activation
{
    public class LinearFunction : IActivationFunction
    {
        public double Function(double x)
        {
            return x;
        }

        public double Derivative(double x)
        {
            return 1;
        }
    }
}
